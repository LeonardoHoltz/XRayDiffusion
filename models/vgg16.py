import torch
from torch import nn, optim
from torchvision import models, transforms
import torchmetrics.classification
import lightning as L

class VGG16(L.LightningModule):
  def __init__(self, num_classes=1, image_size=(224,224), learning_rate=0.005) -> None: 
    super().__init__()

    self.num_classes = num_classes
    self.learning_rate = learning_rate
    self.image_size = image_size

    w,h = self.image_size
    model = models.vgg16()
    for params in model.parameters():
      params.requires_grad = False
    model.requires_grad = False
    self.vgg16 = model.features
    self.avgpool = model.avgpool

    self.classifier = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(7*7*512, 1),
      nn.Sigmoid())

    self.loss_fn = nn.BCELoss()

    # Metrics
    self.accuracy = torchmetrics.Accuracy(
        task="binary", num_classes=1
    )
    self.recall = torchmetrics.Recall(task="binary", num_classes=1)
    self.precision = torchmetrics.Precision(
        task="binary", num_classes=1
    )
    self.f1_score = torchmetrics.F1Score(task="binary", num_classes=1)

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
      

  def forward(self, x):
    x = x.expand(-1, 3, -1, -1)
    out = self.vgg16(x)
    out = self.avgpool(out)
    out = out.reshape(out.size(0), -1)
    out = self.classifier(out)
    out = out.reshape(-1)
    return out

  def _compute_metrics(self, scores, y, mode="train"):
    metrics_dict = {}
    metrics_dict[mode + "/accuracy"] = self.accuracy(scores, y)
    metrics_dict[mode + "/recall"] = self.recall(scores, y)
    metrics_dict[mode + "/precision"] = self.precision(scores, y)
    metrics_dict[mode + "/f1_score"] = self.f1_score(scores, y)
    return metrics_dict

  def show_epoch_results(self, metrics_dict, mode="train") -> None:
    print_result = f"{mode} results:"
    for key, value in metrics_dict.items():
      print_result += f" {key}: {value:.4f} |"
    print(print_result)

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
    y = y.float()
    scores = self.forward(x)
    loss = self.loss_fn(scores, y)
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

    # Log metrics for early stopping
    self.log("val_accuracy", metrics["val/accuracy"])

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