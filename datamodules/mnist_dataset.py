import torchvision.datasets as datasets
import torchvision.transforms as transforms
import lightning as L
from torch.utils.data import DataLoader, random_split

def preprocess(x, y):
    return x.view(-1, 1, 28, 28).to("cuda"), y.to("cuda")

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))

class MnistDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, device="cuda") -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
    
    def prepare_data(self) -> None:
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)
    
    def setup(self, stage):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False,
        )
        self.train_ds, self.val_ds = random_split(entire_dataset, [0.8, 0.2])
        self.test_ds = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False,
        )
    
    def train_dataloader(self):
        return WrappedDataLoader(DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        ), preprocess)
    
    def val_dataloader(self):
        return WrappedDataLoader(DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        ), preprocess)
    
    def test_dataloader(self):
        return WrappedDataLoader(DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        ), preprocess)