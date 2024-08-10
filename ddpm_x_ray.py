import os
import shutil
import tempfile
import time

#from collections.abc import Callable, Sequence
from typing import Callable
from functools import partial

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai import transforms
from torch.nn.modules import Module
import torchvision.transforms as t_transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, CSVDataset, PersistentDataset, pad_list_data_collate
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.networks.nets import SPADEAutoencoderKL, SPADEDiffusionModelUNet

from datasets import load_dataset # Hugging Face
from datamodules.chest_x_ray_dataset import ChestXRayDataModule
import pdb
import shutil
from sklearn.model_selection import train_test_split as sk_train_test_split
from PIL import Image
import config


class ClassConditioningDiffusionInferer(DiffusionInferer):
    def __init__(self, scheduler: Module) -> None:
        super().__init__(scheduler)
    
    def __call__(
        self, inputs,
        diffusion_model,
        noise,
        timesteps,
        classes,
        condition=None,
        mode="crossattn",
        seg= None,
    ):
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        noisy_image = self.scheduler.add_noise(original_samples=inputs, noise=noise, timesteps=timesteps)
        if mode == "concat":
            noisy_image = torch.cat([noisy_image, condition], dim=1)
            condition = None
        diffusion_model = (
            partial(diffusion_model, seg=seg)
            if isinstance(diffusion_model, SPADEDiffusionModelUNet)
            else diffusion_model
        )
        prediction = diffusion_model(x=noisy_image, timesteps=timesteps, class_labels=classes, context=condition)

        return prediction

    @torch.no_grad()
    def sample(
        self,
        input_noise,
        diffusion_model,
        class_label,
        scheduler=None,
        save_intermediates=False,
        intermediate_steps=100,
        conditioning=None,
        mode="crossattn",
        verbose=True,
        seg=None,
    ):
        has_tqdm = True
        if mode not in ["crossattn", "concat"]:
            raise NotImplementedError(f"{mode} condition is not supported")

        if not scheduler:
            scheduler = self.scheduler
        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # 1. predict noise model_output
            diffusion_model = (
                partial(diffusion_model, seg=seg)
                if isinstance(diffusion_model, SPADEDiffusionModelUNet)
                else diffusion_model
            )
            if mode == "concat":
                model_input = torch.cat([image, conditioning], dim=1)
                model_output = diffusion_model(
                    model_input, timesteps=torch.Tensor((t,)).to(input_noise.device), class_labels=class_label, context=None
                )
            else:
                model_output = diffusion_model(
                    image, timesteps=torch.Tensor((t,)).to(input_noise.device), class_labels=class_label, context=conditioning
                )

            # 2. compute previous image: x_t -> x_t-1
            image, _ = scheduler.step(model_output, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)
        if save_intermediates:
            return image, intermediates
        else:
            return image

def main():
    print_config()

    # ## Setup data directory
    #
    # You can specify a directory with the MONAI_DATA_DIRECTORY environment variable.
    # This allows you to save results and reuse downloads.
    # If not specified a temporary directory will be used.
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)

    # ## Set deterministic training for reproducibility
    seed = 69
    set_determinism(seed)

    ## Setup X Ray Dataset
    
    # Train
    
    datamodule = ChestXRayDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    datamodule.set_training_mode('diffusion')
    datamodule.prepare_data()
    datamodule.setup('fit')
    diffusion_dataloader = datamodule.train_dataloader()
    diffusion_val_dataloader = datamodule.val_dataloader()

    # ### Visualisation of the training images
    if True:
        check_data = first(diffusion_dataloader)
        
        print(check_data[0][0].max())
        print(check_data[0][0].min())
        plt.figure("training images", (12, 6))
        plt.axis("off")
        plt.tight_layout()
        plt.subplot(1, 4, 1)
        plt.imshow(check_data[0][0].permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        plt.subplot(1, 4, 2)
        plt.imshow(check_data[0][1].permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        plt.subplot(1, 4, 3)
        plt.imshow(check_data[0][2].permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        plt.subplot(1, 4, 4)
        plt.imshow(check_data[0][3].permute(1, 2, 0), vmin=0, vmax=1, cmap="gray")
        plt.savefig("dataset_example.jpg")
    
    # ### Define network, scheduler, optimizer, and inferer
    # At this step, we instantiate the MONAI components to create a DDPM, the UNET, the noise scheduler, and the inferer used for training and sampling. We are using
    # the original DDPM scheduler containing 1000 timesteps in its Markov chain, and a 2D UNET with attention mechanisms
    # in the 2nd and 3rd levels, each with 1 attention head.

    device = torch.device("cuda")

    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        num_channels=(128, 256, 256, 256),
        attention_levels=(False, False, False, True),
        num_res_blocks=1,
        num_head_channels=256,
        num_class_embeds=2, # It seems that the inferer has its own way to include the class condition, but let's use this one
    )
    model.to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    inferer = ClassConditioningDiffusionInferer(scheduler)

    # ### Model training
    # Here, we are training our model for 75 epochs (training time: ~50 minutes).
    #
    # If you would like to skip the training and use a pre-trained model instead, set `use_pretrained=True`.
    # This model was trained using the code in `MonaiGenerativeModels/tutorials/generative/distributed_training/ddpm_training_ddp.py`

    use_pretrained = False

    if use_pretrained:
        #model = torch.hub.load("marksgraham/pretrained_generative_models:v0.2", model="ddpm_2d", verbose=True).to(device)
        path = 'checkpoint/model.pt'
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_loss_list = checkpoint['epoch_loss_list']
    else:
        n_epochs = 75
        val_interval = 5
        epoch_loss_list = []
        val_epoch_loss_list = []
        
        scaler = GradScaler()
        total_start = time.time()
        for epoch in range(n_epochs):
            model.train()
            epoch_loss = 0
            progress_bar = tqdm(enumerate(diffusion_dataloader), total=len(diffusion_dataloader), ncols=70)
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                torch.cuda.empty_cache()
                images = batch[0].to(device)
                labels = batch[1].to(device)
                
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    # Generate random noise
                    noise = torch.randn_like(images).to(device)

                    # Create timesteps
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()

                    # Get model prediction
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, classes=labels, timesteps=timesteps)

                    loss = F.mse_loss(noise_pred.float(), noise.float())
                
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

            epoch_loss_list.append(epoch_loss / (step + 1))
            
            torch.cuda.empty_cache()
            
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch_loss_list': epoch_loss_list,
                },
                f'checkpoints/diffusion_model_epoch_{epoch}.pt'
            )
            
            if (epoch + 1) % val_interval == 0:
                model.eval()
                val_epoch_loss = 0
                for step, batch in enumerate(diffusion_val_dataloader):
                    images = batch[0].to(device)
                    labels = batch[1].to(device)
                    with torch.no_grad():
                        with autocast(enabled=True):
                            noise = torch.randn_like(images).to(device)
                            timesteps = torch.randint(
                                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                            ).long()
                            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, classes=labels, timesteps=timesteps)
                            val_loss = F.mse_loss(noise_pred.float(), noise.float())

                    val_epoch_loss += val_loss.item()
                    progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
                val_epoch_loss_list.append(val_epoch_loss / (step + 1))

                # Sampling image during training
                noise = torch.randn((1, 1, 224, 224))
                noise = noise.to(device)
                scheduler.set_timesteps(num_inference_steps=1000)
                label_0 = torch.tensor([0]).to(device)
                label_1 = torch.tensor([1]).to(device)
                with autocast(enabled=True):
                    image_0 = inferer.sample(input_noise=noise, diffusion_model=model, class_label=label_0, scheduler=scheduler)
                    image_1 = inferer.sample(input_noise=noise, diffusion_model=model, class_label=label_1, scheduler=scheduler)

                plt.figure(figsize=(2, 2))
                plt.imshow(image_0[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
                plt.tight_layout()
                plt.axis("off")
                plt.show()
                plt.savefig(f"sample_class_0_epoch_{epoch}.jpg")
                
                plt.figure(figsize=(2, 2))
                plt.imshow(image_1[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
                plt.tight_layout()
                plt.axis("off")
                plt.show()
                plt.savefig(f"sample_class_1_epoch_{epoch}.jpg")

        total_time = time.time() - total_start
        print(f"train completed, total time: {total_time}.")

    # ### Learning curves
    if not use_pretrained:
        plt.style.use("seaborn-v0_8")
        plt.title("Learning Curves", fontsize=20)
        plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
        plt.plot(
            np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
            val_epoch_loss_list,
            color="C1",
            linewidth=2.0,
            label="Validation",
        )
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.show()

    # ### Plotting sampling process along DDPM's Markov chain
    model.eval()
    noise = torch.randn((1, 1, 64, 64))
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    with autocast(enabled=True):
        image, intermediates = inferer.sample(
            input_noise=noise, diffusion_model=model, scheduler=scheduler, save_intermediates=True, intermediate_steps=100
        )

    chain = torch.cat(intermediates, dim=-1)

    plt.style.use("default")
    plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
    plt.tight_layout()
    plt.axis("off")
    plt.show()


    # ### Cleanup data directory
    #
    # Remove directory if a temporary was used.
    if directory is None:
        shutil.rmtree(root_dir)

if __name__ == "__main__":
    main()
