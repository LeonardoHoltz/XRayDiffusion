from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from model import MnistSimpleModel
from models.vgg16 import VGG16
from datamodules.chest_x_ray_dataset import ChestXRayDataModule
import config

def main():
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = TensorBoardLogger("tb_logs, ")
    # Datamodule
    datamodule = ChestXRayDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        device=device,
    )
    
    # Initialize network
    model = VGG16(1, datamodule.image_shape).to(device)

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
            EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=5, mode="max"),
        ],
    )
    
    # Training and evaluation
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)

    # TODO: Qualitative prediction results
    
    torch.save(model.state_dict(), "weights/vgg16_weights_dataset.pth")
    

if __name__ == "__main__":
    main()
