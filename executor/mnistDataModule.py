from argparse import ArgumentParser
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import pytorch_lightning as pl

class MnistData(pl.LightningDataModule):
    def __init__(self, root = '/home/bitwiz/codeden/data/mnist', batch_size = 512, workers = 8, debug = False, expt = 'cnn', **kw):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.debug = debug
        self.root = root

        if expt == 'cnn':
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            self.train_transform = transforms.Compose([
                transforms.ToTensor()
            ])

            self.val_transform = transforms.Compose([
                transforms.ToTensor()
            ])


    def trainset(self):
        return MNIST(root=self.root, train=True, download = True, transform = self.train_transform)
    
    def valset(self):
        return MNIST(root=self.root, train=False, download = True, transform = self.val_transform)

    @staticmethod
    def add_loader_specific_args(parent_parser):
        parent_parser.add_argument('-b', '--batch_size', type = int, default = 512)
        parent_parser.add_argument('--workers', type = int, default = 8)
        parent_parser.add_argument('--debug', type = int, default = 0)
        parent_parser.add_argument('--mode', type = int, default = 10)

        return parent_parser
    
    def make_loader(self, split):
        if split == 'train':
            transform = self.train_transform
            dataset = self.trainset()
        elif split == 'val':
            transform = self.val_transform
            dataset = self.valset()
        else:
            transform = self.val_transform
            dataset = self.valset()
        
        loader = DataLoader(dataset, batch_size = self.batch_size, num_workers = self.workers)
        
        return loader

    
    def train_dataloader(self):
        return self.make_loader(split='train')
    
    def val_dataloader(self):
        return self.make_loader(split='val')
    
    def test_dataloader(self):
        return self.make_loader(split='val')
