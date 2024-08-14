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

import matplotlib.pyplot as plt

def is_valid_xray(image, threshold=0.5, min_white_ratio=0.15):
    """
    This function is needed since sometimes the model generates black images
    """
    image_array = image.squeeze()
    white_pixels = torch.sum(image_array > threshold)
    white_ratio = white_pixels / image_array.numel()
    print(f"Ratio de {white_ratio}\n")
    return white_ratio >= min_white_ratio


def save_image(image, class_label, image_name):
    image = image.squeeze()
    image_array = image.cpu().numpy() * 255.0
    image_array = image_array.astype(np.uint8)
    image_pil = Image.fromarray(image_array)
    image_pil.save(f"classification_sample/class_{class_label}/{image_name}", format='JPEG')

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
    epoch_loss_list = checkpoint['epoch_loss_list']
    val_epoch_loss_list = checkpoint['val_epoch_loss_list']
    
    n_epochs = 75
    val_interval = 5
    
    # Learning curve plot (Maybe change this for plotly to check loss details)
    if False:
        plt.figure(frameon=False)
        plt.style.use("seaborn-v0_8")
        plt.title("Learning Curves", fontsize=20)
        plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_loss_list, color="C0", linewidth=2.0, label="Train")
        plt.plot(
            np.linspace(val_interval, n_epochs, n_epochs // val_interval - 1),
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
        plt.savefig("learning_curves.jpg")
    
    os.makedirs("classification_sample/class_0", exist_ok=True)
    os.makedirs("classification_sample/class_1", exist_ok=True)
    label_0 = torch.tensor([0]).to(device)
    label_1 = torch.tensor([1]).to(device)
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(111)): # It will be sampled for the classification training, 220 images
            bad_quality_image_class_0 = True
            bad_quality_image_class_1 = True
            # Sample for class 0
            while(bad_quality_image_class_0):
                noise = torch.randn((1, 1, 224, 224))
                noise = noise.to(device)
                scheduler.set_timesteps(num_inference_steps=1000)
                image_0 = inferer.sample(input_noise=noise, diffusion_model=model, class_label=label_0, scheduler=scheduler)
                bad_quality_image_class_0 = not is_valid_xray(image_0)
                if bad_quality_image_class_0:
                    print("Sample com qualidade ruim, tentando de novo")
            save_image(image_0, class_label=0, image_name=f"sample_{i}_class_0.jpg")
            print(f"imagem {i} classe 0 salva.")
                
            # Sample for class 1
            #while(bad_quality_image_class_1):
            #    noise = torch.randn((1, 1, 224, 224))
            #    noise = noise.to(device)
            #    scheduler.set_timesteps(num_inference_steps=1000)
            #    image_1 = inferer.sample(input_noise=noise, diffusion_model=model, class_label=label_1, scheduler=scheduler)
            #    bad_quality_image_class_1 = not is_valid_xray(image_1)
            #    if bad_quality_image_class_1:
            #        print("Sample com qualidade ruim, tentando de novo")
            #save_image(image_1, class_label=1, image_name=f"sample_{i}_class_1.jpg")
            #print(f"imagem {i} classe 1 salva.")
    print('FIM')

if __name__ == "__main__":
    main()
