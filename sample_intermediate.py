import os
import torch
import lightning as L

from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from datamodules.chest_x_ray_dataset import ChestXRayDataModule
from ddpm_x_ray import ClassConditioningDiffusionInferer
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.amp import autocast

import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    
    path = 'checkpoints/diffusion_model_epoch_74.pt'
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # ### Plotting sampling process along DDPM's Markov chain
    model.eval()
    noise = torch.randn((1, 1, 224, 224))
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    label_0 = torch.tensor([0]).to(device)
    label_1 = torch.tensor([1]).to(device)
    with autocast(device_type="cuda", enabled=True):
        image_0, intermediates_0 = inferer.sample(
            input_noise=noise, diffusion_model=model, class_label=label_0, scheduler=scheduler, save_intermediates=True, intermediate_steps=200
        )
        image_1, intermediates_1 = inferer.sample(
            input_noise=noise, diffusion_model=model, class_label=label_1, scheduler=scheduler, save_intermediates=True, intermediate_steps=200
        )

    chain_0 = torch.cat(intermediates_0, dim=-1)
    chain_1 = torch.cat(intermediates_1, dim=-1)

    plt.figure(frameon=False)
    plt.style.use("default")
    plt.imshow(chain_0[0, 0].cpu(), vmin=0, vmax=1, cmap="gray", aspect='auto')
    plt.axis("off")
    plt.savefig("chain_0.jpg")

    plt.figure(frameon=False)
    plt.style.use("default")
    plt.imshow(chain_1[0, 0].cpu(), vmin=0, vmax=1, cmap="gray", aspect='auto')
    plt.axis("off")
    plt.savefig("chain_1.jpg")
    

if __name__ == "__main__":
    main()
