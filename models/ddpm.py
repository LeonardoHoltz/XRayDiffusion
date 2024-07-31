import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import lightning as L

def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class LightningDDPM(L.LightningModule):
    # Here we have some changes:
    # No need to worry about the devices, since this is already taken care of by the trainer
    
    # We are using the notation of the understanding deep learning book
    def __init__(
        self,
        nn_model,
        betas,
        n_T,
        learning_rate,
        drop_prob=0.1
    ):
        
        super().__init__()
        self.nn_model = nn_model
        self.learning_rate = learning_rate
        self.n_T = n_T
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)
        
        # log outputs
        self.training_loss = []
    
    # Optimizer
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    # Context mask will need to be defined only in the steps
    # training step: will use the dropouit_prob
    # test, predict and validation will use a zero mask
    def forward(self, z_t, y, t, context_mask):
        """_summary_

        Args:
            x_t: noised image x based on the time t. (the greater the t, the noiser is the image)
            I think the shape is (batch_size, channel, width, height)
            
            y: class label of the image to be forwarded
            
            t: ratio of current time t over the total number of times
            (near 1 means we want a more noisy image, near 0)
            
            context_mask (_type_): _description_
        """
        return self.nn_model(z_t, y, t, context_mask)
    
    def diffusion_encoder_at_level_t(self, t, x, noise):
        # Uses broadcasting (addition of the None indices to match the shape of x and the noise)
        z_t = (
            self.sqrtab[t, None, None, None] * x
          + self.sqrtmab[t, None, None, None] * noise
        )
        return z_t
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # Creates the noise to be used with the same size as x
        # (uses a normal distribution as expected)
        noise = torch.randn_like(x)
        
        # Choose a random t based on uniform distribution
        random_t = torch.randint(1, self.n_T+1, (x.shape[0],))
        
        # Applies the diffusion kernel
        # (obtain the noisy image at the Tth level)
        z_t = self.diffusion_encoder_at_level_t(random_t, x, noise)
        
        # adds context mask based on the dropout probability
        context_mask = torch.bernoulli(torch.zeros_like(y)+self.drop_prob).to(self.device)
        
        # Obtains the ratio of the randomized t
        random_t_ratio = random_t / self.n_T
        
        # forwards the model
        scores = self.forward(z_t, y, random_t_ratio, context_mask)
        
        # Return the MSE loss between noise and the predicted noise
        loss = self.loss_mse(noise, scores)
        self.training_loss.append(loss)

    def show_epoch_results(self, metrics_dict, mode="train") -> None:
        print_result = f"{mode} results:"
        for key, value in metrics_dict.items():
            print_result += f" {key}: {value:.4f} |"
        print(print_result)
    
    # Epoch callbacks
    def on_train_epoch_end(self) -> None:
        # Concat results
        loss = torch.stack(self.training_loss).mean()

        # clean outputs
        self.training_loss.clear()

        # Compute and log metrics
        metrics = {}
        metrics["train/loss"] = loss
        self.log_dict(
            metrics, logger=self.logger, on_step=False, on_epoch=True, prog_bar=False
        )
        self.show_epoch_results(metrics)
        torch.cuda.empty_cache()
    
    def _sampling_step(self, size, y, t=0, guide_w=0.0):
        """
        Step in which the testing and predict step (and probably the validation,
        but we will not use it for now) will use to sample one image from the diffusion model
        
        Currently not doubling the sample to use half as context free
        
        Args:
            size: Shape of the image to be sampled
            y: Class label of the image to be sampled
            t: Level t of image to be sampled
        """
        y = torch.tensor([y])
        
        # Set initial noise
        z_T = torch.randn(1, *size)
        
        # Don't drop context at test time
        context_mask = torch.zeros_like(y)
        
        t_ratio = torch.tensor([t / self.n_T])
        t_ratio = t_ratio.repeat(1, 1, 1, 1)
        
        # double the batch
        y = y.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[1:] = 1 # makes second half of batch context free
        z_T = z_T.repeat(2, 1, 1, 1)
        t_ratio = t_ratio.repeat(2, 1, 1, 1)
        
        z = torch.randn(1, *size) if t > 1 else 0
        
        eps = self.forward(z_T, y, t_ratio, context_mask)
        eps1 = eps[:1]
        eps2 = eps[1:]
        eps = (1+guide_w) * eps1 - guide_w * eps2
        z_T = z_T[:1]
        sampled_image = (
            self.oneover_sqrta[t] * (z_T - eps * self.mab_over_sqrtmab[t]) +
            self.sqrt_beta_t[t] * z
        )
        
        return sampled_image
        
        
        
    
    # TODO: Probably this type of function should be defined at a custom trainer, but im not sure of it
    #def sample_images(self, n_sample, size, guide_w):
        
