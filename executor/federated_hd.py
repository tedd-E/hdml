from argparse import ArgumentParser
import numpy as np
import pickle
import copy
import gc

import torch
import torch.nn as nn
import torch_hd.hdlayers as hd
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from pl_bolts.models.self_supervised import SimCLR
from cifarDataModule import CifarData

import wandb


class federated_framework:
    def __init__(
        self, model, data_splits, test_data, nclients=10, fraction=0.5, nrounds=20,
        local_epochs=5, local_batch_size=10, cpu=False, lr=1, debug=False,
        scale_lr=0.99, log=False, cuda=0, iid=True, **kwargs
    ):

        self.lr = lr
        self.iid = iid
        self.encoder, self.classifier = copy.deepcopy(model)
        self.classifier.alpha = lr
        nclients = len(data_splits)
        self.clients = [copy.deepcopy(self.classifier)
                        for _ in range(nclients)]
        print("=> Initialized clients")

        self.nclients = nclients
        self.train_data = data_splits
        self.test_data = test_data
        print("=> Initialized data")

        self.E = local_epochs
        self.B = local_batch_size
        self.C = fraction
        self.device = 'cpu' if cpu else 'cuda:' + str(cuda)
        self.debug = debug
        self.nrounds = nrounds
        self.log = log

        if log:
            wandb.init(project='federated')

    def train(self, target_acc=0.85):
        test_acc = 0.0
        run_counter = 0
        results = [0]
        rounds = [0]

        print("=> Begining training")

        while run_counter < self.nrounds:
            run_counter += 1

            num = np.ceil(self.C * self.nclients).astype(np.int)
            choices = np.arange(0, self.nclients)
            chosen = np.random.choice(choices, size=(num,), replace=True)

            for client in chosen:
                self.fit(client)

            updated_params = self.classifier.class_hvs.clone().to(self.device)
            for client in chosen:
                updated_params += self.clients[client].class_hvs
            updated_params /= num

            self.classifier.class_hvs = nn.Parameter(
                updated_params, requires_grad=False)
            if not self.iid:
                self.classifier.alpha = self.lr + (1 / (1 + run_counter))

            self.broadcast()

            test_acc = self.test()
            results.append(test_acc)
            rounds.append(run_counter)

            if self.log:
                wandb.log({
                    'test_acc': test_acc,
                    'round': run_counter
                })

            print("\t=> Run: {} test accuracy: {}".format(run_counter, test_acc))

        history = {
            'rounds': rounds,
            'acc': results
        }

        if self.log:
            wandb.finish()

        return history

    def fit(self, client):
        loader = DataLoader(
            self.train_data[client], batch_size=self.B, shuffle=True)
        self.clients[client] = self.clients[client].to(self.device)
        encoder = self.encoder.to(self.device)
        self.clients[client].train()

        for epoch in range(self.E):
            overall_acc = 0
            for idx, batch in enumerate(loader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device).type(torch.int)
                x = encoder(x)
                y_hat = self.clients[client](x, y)
                _, y_hat = torch.max(y_hat, dim=1)
                acc = accuracy(y_hat, y)
                overall_acc += acc

            overall_acc /= (idx + 1)
            self.clients[client].oneshot = True

        if self.debug:
            print("\t\t=> client_id: {} accuracy: {}".format(client, overall_acc))

    def broadcast(self):
        self.classifier.oneshot = True
        self.clients = [copy.deepcopy(self.classifier)
                        for _ in range(self.nclients)]

    def test_client(self, client):
        loader = DataLoader(self.test_data, batch_size=128, shuffle=False)
        encoder = self.encoder.to(self.device)
        self.clients[client] = self.clients[client].to(self.device)
        self.clients[client].train()

        overall_acc = 0
        for idx, batch in enumerate(loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            x = encoder(x)
            y_hat = self.clients[client](x)
            _, y_hat = torch.max(y_hat, dim=1)
            acc = accuracy(y_hat, y)
            overall_acc += acc

        overall_acc /= (idx + 1)

        return overall_acc

    def test(self):
        loader = DataLoader(self.test_data, batch_size=128, shuffle=False)
        encoder = self.encoder.to(self.device)
        classifier = self.classifier.to(self.device)
        classifier.eval()

        overall_acc = 0
        for idx, batch in enumerate(loader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            x = encoder(x)
            y_hat = classifier(x)
            _, y_hat = torch.max(y_hat, dim=1)
            acc = accuracy(y_hat, y)
            overall_acc += acc

        overall_acc /= (idx + 1)

        return overall_acc


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = CifarData.add_loader_specific_args(parser)
    parser.add_argument('--cuda', type=int, default=0)

    args = parser.parse_args()

    weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/simclr-cifar10-v1-exp12_87_52/epoch%3D960.ckpt'
    net = SimCLR.load_from_checkpoint(
        weight_path, strict=False, dataset='cifar10', maxpool1=False, first_conv=False, input_height=32)
    net.freeze()

    encoder = nn.Sequential(
        net,
        hd.hd_rp_encoder(2048, 10000)
    )

    classifier = hd.hd_classifier(10, 10000)

    model = (encoder, classifier)
    print("=> Created model")
    data = CifarData(root='/data/cifar', batch_size=512)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    data.train_transform = transform
    data.val_transform = transform

    dev = args.cuda
    debug_flag = False

    c_range = np.linspace(0.1, 1.0, num=10)

    splits = split_cifar_niim(data)

    fed = federated_framework(model, splits, data.valset(
    ), E=2, B=10, C=0.1, nclients=100, nrounds=70, device=dev, debug=debug_flag)
    fed.train()

    '''
    for c in c_range:
        fed = federated_framework(model, splits, data.valset(), E = 1, B = 10, C = c, nclients = 100, debug = True)
        history_e1 = fed.train()
        logs_e1.append(history_e1)

        fed = federated_framework(model, splits, data.valset(), E = 5, B = 10, C = c, nclients = 100, debug = True)
        history_e5 = fed.train()
        logs_e5.append(history_e5)

    save_data = (logs_e1, logs_e5)
    pickle.dump(save_data, open('/codeden/federated/logs/averaged/test_c_e1_e5_niid.p', 'wb'))
    '''
