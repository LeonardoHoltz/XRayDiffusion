from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from models.vgg16 import *
import time

image_w = 400
image_h = 400
batch_size = 2
file_path = f"weights/vgg16_weights_{image_w}x{image_h}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading dataset...")
ds = load_dataset("AiresPucrs/chest-xray", split="train")
split = ds.train_test_split()
train_ds = split["train"]
test_ds = split["test"]

count_train_normal = 0
for data in train_ds['label']:
  if data == 0:
    count_train_normal += 1

count_test_normal = 0
for data in test_ds['label']:
  if data == 0:
    count_test_normal += 1

print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
print(f"Train label distribution: {format(count_train_normal/len(train_ds), '.00%')} normal, {format(1 - count_train_normal/len(train_ds), '.00%')} pneumonia")
print(f"Test label distribution: {format(count_test_normal/len(test_ds), '.00%')} normal, {format(1 - count_test_normal/len(test_ds), '.00%')} pneumonia")


print("Building data loader...")

transform_resize = Compose([
  Resize((image_w, image_h)),
  ToTensor()
])
transform_pad = Compose([
  CenterCrop((image_w, image_h)),
  ToTensor()
])
def apply_transform(ds):
  ds["image"] = [transform_pad(image.convert("RGB")) if image.size[0] < image_w and image.size[1] < image_h else transform_resize(image.convert("RGB")) for image in ds["image"]]
  ds["label"] = [torch.tensor([1.0,0.0]) if label == 0 else torch.tensor([0.0,1.0]) for label in ds["label"]]
  return ds

train_ds.set_format("torch", device=device, columns=['image','label'])
train_ds.set_transform(apply_transform)

test_ds.set_format("torch", device=device, columns=['image','label'])
test_ds.set_transform(apply_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

for batch in train_loader:
  for i in range(batch_size):
    img = batch['image'][i].squeeze().permute(1, 2, 0)
    label = batch['label'][i]
    plt.subplot(4, 4, i+1)
    plt.gca().set_title(f"Label: {'NORMAL' if label[0] == 0 else 'PNEUMONIA'}")
    plt.imshow(img)
  plt.show()
  break

print("Loading model...")
model = VGG16(2)
model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

print("Training model...")
num_epochs = 5
total_batches = len(train_loader)

with torch.no_grad():
  print("Evaluating model...")
  correct = 0
  total = 0
  for batch in test_loader:
    images = batch['image'].to(device)
    labels = batch['label'].to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == torch.max(labels, 1)[1]).sum().item()
  print(f"Accuracy before training: {100*correct/total:.2f}%")

for epoch in range(num_epochs):
  t0 = time.time()
  for i, batch in enumerate(train_loader):
    images = batch['image'].to(device)
    labels = batch['label'].to(device)

    optimizer.zero_grad()
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()

    if i % 40 == 0 or i+1 == total_batches:
      t1 = time.time()
      print(f"Epoch {epoch+1}/{num_epochs}, step {i+1}/{total_batches}, {t1-t0:.2f}s, loss = {loss.item()}")
      t0 = time.time()
  
  with torch.no_grad():
    print("Evaluating model...")
    correct = 0
    total = 0
    for batch in test_loader:
      images = batch['image'].to(device)
      labels = batch['label'].to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == torch.max(labels, 1)[1]).sum().item()
    print(f"Accuracy on test set: {100*correct/total:.2f}%")

torch.save(model.state_dict(), file_path)