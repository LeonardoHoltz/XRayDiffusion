import torch
import torch.nn as nn
from models.simple_cnn import SimpleCNN
from datamodules.chest_x_ray_dataset import ChestXRayDataModule
import config
from tqdm import tqdm
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import wandb


def main():
    logger = WandbLogger(project='XRayClassifier', log_model='all')
    wandb.login()
    
    # 'Reduce to merge': reduce training data: 50% sampled and 50% original (TRAINED)
    # 'Reduce': reduce training data and use only original training data
    # 'Use Sample Only': Se only sampled data
    # 'Merge Only': Only concatenates original with sampled (TRAINED)
    # else: Use training data normally (TRAINED)
    datamodule = ChestXRayDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        reduce_train="Use Sample Only", 
    )
    datamodule.set_training_mode('classification')
    
    model = SimpleCNN(num_classes=2, learning_rate=0.0001)
    
    # Trainer
    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        logger=logger,
        log_every_n_steps=0,
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
