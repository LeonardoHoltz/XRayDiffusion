import torch
import lightning as L

from models.ddpm import LightningDDPM
from models.unet import ContextUnet
from datasets.dataset import ChestXRayDataModule
import config

import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load network from checkpoint
    model = LightningDDPM.load_from_checkpoint("checkpoints/lightning_logs/version_0/checkpoints/epoch=24-step=9150.ckpt")
    model.eval()
    model.to(device)

    normal_image = model._sampling_step((1, 224, 224), 0, t=0).cpu().detach()
    print(normal_image.shape)
    normal_image = torch.squeeze(normal_image, 0)
    print(normal_image.shape)
    plt.imshow(normal_image.permute(1, 2, 0))
    plt.savefig("example_epoch_24.png")

if __name__ == "__main__":
    main()
