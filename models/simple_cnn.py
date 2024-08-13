import torch
from torch import nn, optim
import torch.nn.functional as F
import torchmetrics.classification
import lightning as L

class SimpleCNN(L.LightningModule):
    def __init__(self, num_classes, learning_rate) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Modules
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=128 * 28 * 28, out_features=256) # considering input of 224 with 3 convolutions
        self.fc2 = nn.Linear(in_features=256, out_features=1)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flat = nn.Flatten()
        
        self.loss_fn = nn.BCELoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(
            task="binary", num_classes=num_classes
        )
        self.recall = torchmetrics.Recall(task="binary", num_classes=num_classes)
        self.precision = torchmetrics.Precision(
            task="binary", num_classes=num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="binary", num_classes=num_classes)

        # Log Outputs
        self.train_scores = []
        self.train_y_trues = []

        self.val_scores = []
        self.val_y_trues = []

        self.test_scores = []
        self.test_y_trues = []

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

    def _compute_metrics(self, scores, y, mode="train"):
        metrics_dict = {}
        metrics_dict[mode + "/accuracy"] = self.accuracy(scores.squeeze(), y)
        metrics_dict[mode + "/recall"] = self.recall(scores.squeeze(), y)
        metrics_dict[mode + "/precision"] = self.precision(scores.squeeze(), y)
        metrics_dict[mode + "/f1_score"] = self.f1_score(scores.squeeze(), y)
        return metrics_dict

    def show_epoch_results(self, metrics_dict, mode="train") -> None:
        print_result = f"{mode} results:"
        for key, value in metrics_dict.items():
            print_result += f" {key}: {value:.4f} |"
        print(print_result)

    ## Forward functions
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        
        return x

    # Steps
    def training_step(self, batch, batch_idx):
        # when is desirable to train the model
        scores, y, loss = self._common_step(batch, batch_idx)
        self.train_scores.append(scores)
        self.train_y_trues.append(y)
        self.train_losses.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # When is desirable to validate the model on unseen data during training
        scores, y, loss = self._common_step(batch, batch_idx)
        self.val_scores.append(scores)
        self.val_y_trues.append(y)
        self.val_losses.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        # When is desirable to evaluate the model on unseen data
        scores, y, loss = self._common_step(batch, batch_idx)
        self.test_scores.append(scores)
        self.test_y_trues.append(y)
        self.test_losses.append(loss)
        return scores, loss

    def predict_step(self, batch, batch_idx):
        # When is desirable to know the final result
        x, y = batch
        scores = self.forward(x)
        predictions = torch.argmax(scores, dim=1)
        return predictions

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores.squeeze(), y.float())
        return scores, y, loss

    # Optimizer
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    # Epoch callbacks
    def on_train_epoch_end(self) -> None:
        # Concat results
        scores = torch.cat(self.train_scores)
        y = torch.cat(self.train_y_trues)
        loss = torch.stack(self.train_losses).mean()

        # clean outputs
        self.train_scores.clear()
        self.train_y_trues.clear()
        self.train_losses.clear()

        # Compute and log metrics
        metrics = self._compute_metrics(scores, y)
        metrics["train/loss"] = loss
        self.log_dict(
            metrics, logger=self.logger, on_step=False, on_epoch=True, prog_bar=False
        )
        self.show_epoch_results(metrics)

    def on_validation_epoch_end(self) -> None:
        # Concat results
        scores = torch.cat(self.val_scores)
        y = torch.cat(self.val_y_trues)
        loss = torch.stack(self.val_losses).mean()

        # clean outputs
        self.val_scores.clear()
        self.val_y_trues.clear()
        self.val_losses.clear()

        # Compute and log metrics
        metrics = self._compute_metrics(scores, y, "val")
        metrics["val/loss"] = loss
        if not self.trainer.sanity_checking:
            self.log_dict(
                metrics,
                logger=self.logger,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            if self.running_fit:
                self.show_epoch_results(metrics, mode="val")

    def on_test_epoch_end(self) -> None:
        # Concat results
        scores = torch.cat(self.test_scores)
        y = torch.cat(self.test_y_trues)
        loss = torch.stack(self.test_losses).mean()

        # clean outputs
        self.test_scores.clear()
        self.test_y_trues.clear()
        self.test_losses.clear()

        # Compute and log metrics
        metrics = self._compute_metrics(scores, y, "test")
        metrics["test/loss"] = loss
        self.log_dict(
            metrics, logger=self.logger, on_step=False, on_epoch=True, prog_bar=False
        )

    def on_fit_start(self) -> None:
        self.running_fit = True

    def on_fit_end(self) -> None:
        self.running_fit = False
