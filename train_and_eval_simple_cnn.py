import torch
import torch.nn as nn
from models.simple_cnn import SimpleCNN
from datamodules.chest_x_ray_dataset import ChestXRayDataModule
import config
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary, EarlyStopping

def main():
    
    datamodule = ChestXRayDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )
    datamodule.set_training_mode('classification')
    # Use this when using sampled images from diffusion
    datamodule.use_sampled_data(True)
    
    model = SimpleCNN(num_classes=2, learning_rate=0.0001)
    
    # Trainer
    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[
            RichProgressBar(leave=True), 
            RichModelSummary(),
            EarlyStopping(monitor="val/accuracy", patience=8, mode="max"),
        ],
    )
    
    # Training and evaluation
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)

if __name__ == "__main__":
    main()
