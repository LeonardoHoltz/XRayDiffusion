import os
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from models.ddpm import LightningDDPM
from models.unet import ContextUnet
from datamodules.chest_x_ray_dataset import ChestXRayDataModule
from datamodules.mnist_dataset import MnistDataModule
import config
from torchinfo import summary

def main():
    torch.set_float32_matmul_precision("medium")
    os.makedirs(config.MODEL_CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TensorBoardLogger("tb_logs, ")
    
    # Datamodule
    #datamodule = ChestXRayDataModule(
    #    data_dir=config.DATA_DIR,
    #    batch_size=config.BATCH_SIZE,
    #    num_workers=config.NUM_WORKERS,
    #    mode='diffusion'
    #)
    datamodule = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    
    # Initialize network
    internal_model = ContextUnet(in_channels=1, n_feat=config.N_FEATURES, n_classes=config.NUM_CLASSES).to(device)
    
    model = LightningDDPM(
        nn_model=internal_model,
        betas=config.BETAS,
        n_T=config.N_T,
        learning_rate=config.LEARNING_RATE,
    ).to(device)
    
    
    # Trainer
    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[RichProgressBar(leave=True), RichModelSummary(max_depth=10), DeviceStatsMonitor()],
        default_root_dir=config.MODEL_CHECKPOINT_DIR,
    )
    
    trainer.fit(model, datamodule)
    

if __name__ == "__main__":
    main()
