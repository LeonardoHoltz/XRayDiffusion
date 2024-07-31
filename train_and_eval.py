from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

from model import MnistSimpleModel
from dataset import ChestXRayDataModule
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
    )
    
    # Initialize network
    model = MnistSimpleModel(
        input_size=config.INPUT_SIZE,
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE
    ).to(device)

    # Trainer
    trainer = L.Trainer(
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[RichProgressBar(leave=True), RichModelSummary()],
    )
    
    # Training and evaluation
    trainer.fit(model, datamodule)
    trainer.validate(model, datamodule)
    trainer.test(model, datamodule)

    # TODO: Qualitative prediction results
    

if __name__ == "__main__":
    main()
