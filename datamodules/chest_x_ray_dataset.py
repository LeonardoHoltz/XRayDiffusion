#import torchvision.datasets as datasets
from enum import Enum
from datasets import load_dataset
import torchvision.transforms as transforms
import lightning as L
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split

class ChestXRayDataset(Dataset):
    def __init__(self, dataset, target_shape, transform_resize=None, transform_padding=None):
        self.dataset = dataset
        self.transform_resize = transform_resize
        self.transform_padding = transform_padding
        self.target_shape = target_shape
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]['image']
        label = self.dataset[idx]['label']
        if self.transform_resize and self.transform_padding:
            if image.size[-2] < self.target_shape[0] and image.size[-1] < self.target_shape[1]:
                image = self.transform_padding(image)
            else:
                image = self.transform_resize(image)
        return image, label

class ChestXRayDatasetPerLabel(ChestXRayDataset):
    def __init__(self, dataset, label, transform=None):
        super().__init__(dataset, transform)
        self.indices = [i for i in range(len(dataset)) if dataset['label'][i] == label]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return super().__getitem__(real_idx)


class ChestXRayDataModule(L.LightningDataModule):
    
    def __init__(self, data_dir, batch_size, num_workers, mode='classification', device='cuda'):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_shape = (224, 224)
        self.training_mode = mode
        self.device = device

    def prepare_data(self) -> None:
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        load_dataset(self.data_dir)

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        entire_dataset = load_dataset(self.data_dir)
        entire_dataset['train']
        
        # Transform to be used
        transform_resize = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Resize((self.image_shape[0], self.image_shape[1])),
        ])
        transform_padding = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.CenterCrop((self.image_shape[0], self.image_shape[1])),
        ])
        
        # The split will be conducted as:
        # 50% for the diffusion model training
        # 30% for the classification model training
        # 10% for the classification model validation
        # 10% for the classification model testing
        diffusion_classification_split = entire_dataset['train'].train_test_split(test_size=0.5, stratify_by_column='label')
        diffusion_dataset = diffusion_classification_split['train']
        classification_dataset = diffusion_classification_split['test']
        
        classification_train_test_split = classification_dataset.train_test_split(test_size=0.2, stratify_by_column='label')
        train_dataset = classification_train_test_split['train']
        test_dataset = classification_train_test_split['test']
        
        # split the train dataset into 90% train and 10% validation
        train_val_split = train_dataset.train_test_split(test_size=0.25, stratify_by_column='label')
        self.diffusion_dataset = ChestXRayDataset(diffusion_dataset, self.image_shape, transform_resize, transform_padding)
        self.train_dataset = ChestXRayDataset(train_val_split['train'], self.image_shape, transform_resize, transform_padding)
        self.val_dataset = ChestXRayDataset(train_val_split['test'], self.image_shape, transform_resize, transform_padding)
        self.test_dataset = ChestXRayDataset(test_dataset, self.image_shape, transform_resize, transform_padding)

    def set_training_mode(self, mode="classification"):
        if mode in ['classification', 'diffusion']:
            self.training_mode = mode
    
    def train_dataloader(self):
        if self.training_mode == 'classification':
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )
        else:
            return DataLoader(
                self.diffusion_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

# Dataset analysis functions (no training)

def get_dataloader_from_hf(dataset_name, batch_size):
    dataset = load_dataset(dataset_name)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    custom_dataset = ChestXRayDataset(dataset['train'], transform)
    return DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

def get_dataloaders_by_label(dataset_name, n_labels, batch_size):
    dataset = load_dataset(dataset_name)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataloaders = []
    for i in range(n_labels):
        custom_dataset = ChestXRayDatasetPerLabel(dataset['train'], i, transform)
        dataloaders.append(DataLoader(custom_dataset, batch_size=batch_size, shuffle=True))
    return dataloaders