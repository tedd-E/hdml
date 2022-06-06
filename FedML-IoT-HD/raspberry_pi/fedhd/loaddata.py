import argparse
import logging
import os
import sys
import time

import numpy as np
import copy

import torch
import torch.nn as nn
import torch_hd.hdlayers as hd
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics.functional import accuracy
import torchvision.transforms as transforms

from pl_bolts.models.self_supervised import SimCLR

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from FedML.fedml_api.distributed.fedhd.fedhd_ModelTrainer import MyModelTrainer
from FedML.fedml_api.distributed.fedhd.fedhd_Trainer import fedHD_Trainer
from FedML.fedml_api.distributed.fedhd.fedhd_ClientManager import FedHDClientManager
from FedML.fedml_api.distributed.fedhd.FedhdAggregator import FedHDAggregator
from FedML.fedml_api.distributed.fedhd.FedhdServerManager import FedHDServerManager

from FedML.fedml_api.data_preprocessing.load_data import load_partition_data
from FedML.fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from FedML.fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10





def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['mnist', 'fashionmnist', 'cifar10'],
                        help='dataset used for training')


    parser.add_argument('--partition_method', type=str, default='iid',
                        choices=['iid', 'noniid'],
                        help='how to partition the dataset on local clients')


    parser.add_argument('--client_num_in_total', type=int, default=8,
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=8,
                        help='number of workers')


    parser.add_argument('--batch_size', type=int, default=100,
                        help='input batch size for training (default: 64)')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')


    parser.add_argument('--epochs', type=int, default=5,
                        help='how many epochs will be trained locally')

    parser.add_argument('--comm_round', type=int, default=100,
                        help='how many round of communications we should use')


    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    # Communication settings
    parser.add_argument('--backend', type=str, default='MQTT',
                        choices=['MQTT', 'MPI'],
                        help='communication backend')
                        
                        
    parser.add_argument('--mqtt_host', type=str, default='127.0.0.1',
                        help='host IP in MQTT')
                        
                        
    parser.add_argument('--mqtt_port', type=int, default=1883,
                        help='host port in MQTT')



    #
    parser.add_argument('--partition_alpha', type=float, default=0.5,
                        help='partition alpha (default: 0.5), used as the proportion'
                             'of majority labels in non-iid in latest implementation')

    parser.add_argument('--partition_secondary', type=bool, default=False,
                        help='True to sample minority labels from one random secondary class,'
                             'False to sample minority labels uniformly from the rest classes except the majority one')

    parser.add_argument('--partition_label', type=str, default='uniform',
                        choices=['uniform', 'normal'],
                        help='how to assign labels to clients in non-iid data distribution')

    parser.add_argument('--data_size_per_client', type=int, default=600,
                        help='input batch size for training (default: 64)')

    parser.add_argument('--D', type=int, default=10000,
                        help='HD encoder dim')




    args = parser.parse_args()
    return args







def load_data(args, dataset_name):
    if dataset_name == "shakespeare":
        logging.info("load_data. dataset_name = %s" % dataset_name)
        client_num, train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = load_partition_data_shakespeare(args.batch_size)
        args.client_num_in_total = client_num
    elif dataset_name == "cifar100" or dataset_name == "cinic10":
        if dataset_name == "cifar100":
            data_loader = load_partition_data_cifar100 # Not tested
        else: # cinic10
            data_loader = load_partition_data_cinic10 # Not tested

        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        data_dir = './../../../data/' + args.dataset
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, data_dir, args.partition_method,
                                args.partition_alpha, args.client_num_in_total,
                                args.batch_size)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))
    elif dataset_name == "mnist" or dataset_name == "fashionmnist" or \
        dataset_name == "cifar10":
        data_loader = load_partition_data
        print(
            "============================Starting loading {}==========================#".format(
                args.dataset))
        data_dir = './../../../data/' + args.dataset
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, \
        class_num = data_loader(args.dataset, data_dir, args.partition_method,
                                args.partition_label, args.partition_alpha, args.partition_secondary,
                                args.client_num_in_total, args.batch_size,
                                args.data_size_per_client)
        print(
            "================================={} loaded===============================#".format(
                args.dataset))
    else:
        raise ValueError('dataset not supported: {}'.format(args.dataset))

    dataset = [train_data_num, test_data_num, train_data_global, test_data_global,
               train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num]
    return dataset









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    print(args)


    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
    train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    projector = hd.RandomProjectionEncoder(2048, args.D)
    
    net = SimCLR.load_from_checkpoint(
    "epoch=960.ckpt", strict=False, dataset='imagenet', maxpool1=False, first_conv=False, input_height=28)
    
    net.freeze()

    print("dataLoaded")
    
    for batch_idx, (x, labels) in enumerate(train_data_local_dict[3]):
    	print(batch_idx)
    	print(x)
    	print(labels)
    	print(x.shape)
    	net(x)
    





    exit(0)





