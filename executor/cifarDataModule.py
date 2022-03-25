from argparse import ArgumentParser
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
import pytorch_lightning as pl

class CifarData(pl.LightningDataModule):
    def __init__(self, root = '/home/bitwiz/codeden/data/cifar', mode = 10, batch_size = 512,
            workers = 8, debug = False, expt = 'cnn', **kw):
        super().__init__()
        self.batch_size = batch_size
        self.workers = workers
        self.debug = debug
        self.mode = 'CIFAR' + str(mode)
        self.root = root

        cifar10_means, cifar10_stds = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        cifar100_means, cifar100_stds = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)

        if mode == 10:
            means = cifar10_means
            stds = cifar10_stds
        elif mode == 100:
            means = cifar100_means
            stds = cifar100_stds
        else:
            print("error")

        if expt == 'cnn':
            self.train_transform = transforms.Compose([
                #transforms.Pad(4),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(means, stds)
            ])

            self.val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds)
            ])
        elif expt == 'hd':
            self.train_transform = transforms.Compose([
                #transforms.Pad(4),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomCrop(32),
                transforms.ToTensor()
            ])

            self.val_transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            print("invalid option")
            exit()


    def trainset(self):
        return eval(self.mode + "(root=self.root, train=True, download = True, transform = self.train_transform)")
    
    def valset(self):
        return eval(self.mode + "(root=self.root, train=False, download = True, transform = self.val_transform)")

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
            dataset = eval(self.mode + "(root=self.root, train=True, transform = transform)")
        elif split == 'val':
            transform = self.val_transform
            dataset = eval(self.mode + "(root=self.root, train=False, transform = transform)")
        else:
            transform = self.val_transform
            dataset = eval(self.mode + "(root=self.root, train=False, transform = transform)")
        
        loader = DataLoader(dataset, batch_size = self.batch_size, num_workers = self.workers)
        
        return loader

    
    def train_dataloader(self):
        return self.make_loader(split='train')
    
    def val_dataloader(self):
        return self.make_loader(split='val')
    
    def test_dataloader(self):
        return self.make_loader(split='val')
