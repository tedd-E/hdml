
import logging
import os
import sys

import argparse
import numpy as np
import torch
import wandb

import torch
import torch.nn as nn
import torch_hd.hdlayers as hd
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchmetrics.functional import accuracy
import torchvision.transforms as transforms

from pl_bolts.models.self_supervised import SimCLR
from cifarDataModule import CifarData


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from FedML.fedml_api.distributed.fedhd.FedhdAggregator import FedAVGAggregator
from FedML.fedml_api.distributed.fedhd.FedhdServerManager import FedHDServerManager
from FedML.fedml_api.distributed.fedhd.fedhd_Triner import MyModelTrainer


from FedML.fedml_api.data_preprocessing.load_data import load_partition_data
from FedML.fedml_api.data_preprocessing.cifar100.data_loader import load_partition_data_cifar100
from FedML.fedml_api.data_preprocessing.cinic10.data_loader import load_partition_data_cinic10
from FedML.fedml_api.data_preprocessing.shakespeare.data_loader import load_partition_data_shakespeare


from FedML.fedml_core.distributed.communication.observer import Observer

from flask import Flask, request, jsonify, send_from_directory, abort


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


    args = parser.parse_args()
    return args


# HTTP server
app = Flask(__name__)
app.config['MOBILE_PREPROCESSED_DATASETS'] = './preprocessed_dataset/'

# parse python script input parameters
parser = argparse.ArgumentParser()
args = add_args(parser)

device_id_to_client_id_dict = dict()


@app.route('/', methods=['GET'])
def index():
    return 'backend service for Fed_mobile'


@app.route('/get-preprocessed-data/<dataset_name>', methods = ['GET'])
def get_preprocessed_data(dataset_name):
    directory = app.config['MOBILE_PREPROCESSED_DATASETS'] + args.dataset.upper() + '_mobile_zip/'
    try:
        return send_from_directory(
            directory,
            filename=dataset_name + '.zip',
            as_attachment=True)

    except FileNotFoundError:
        abort(404)


@app.route('/api/register', methods=['POST'])
def register_device():
    global device_id_to_client_id_dict
    # __log.info("register_device()")
    device_id = request.args['device_id']
    registered_client_num = len(device_id_to_client_id_dict)
    if device_id in device_id_to_client_id_dict:
        client_id = device_id_to_client_id_dict[device_id]
    else:
        client_id = registered_client_num + 1
        device_id_to_client_id_dict[device_id] = client_id

    training_task_args = {"dataset": args.dataset,
                          "data_dir": './../../data/' + args.dataset,
                          "partition_method": args.partition_method,

                          "client_num_per_round": args.client_num_per_round,
                          "client_num_in_total": args.client_num_in_total,

                          "comm_round": args.comm_round,
                          "epochs": args.epochs,

                          "lr": args.lr,

                          "batch_size": args.batch_size,
                          "frequency_of_the_test": args.frequency_of_the_test,

                          'dataset_url': '{}/get-preprocessed-data/{}'.format(
                              request.url_root,
                              client_id-1
                          ),

                          'backend': args.backend,
                          'mqtt_host': args.mqtt_host,
                          'mqtt_port': args.mqtt_port}


    return jsonify({"errno": 0,
                    "executorId": "executorId",
                    "executorTopic": "executorTopic",
                    "client_id": client_id,
                    "training_task_args": training_task_args})


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
        data_dir = './../../data/' + args.dataset
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
        data_dir = './../../data/' + args.dataset
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
    # MQTT client connection
    class Obs(Observer):
        def receive_message(self, msg_type, msg_params) -> None:
            print("receive_message(%s,%s)" % (msg_type, msg_params))

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == 'darwin':
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    logging.info(args)
    


    # GPU 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load data
    dataset = load_data(args, args.dataset)
    [train_data_num, test_data_num, train_data_global, test_data_global,
     train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)


	 net = SimCLR.load_from_checkpoint(
   	"epoch=960.ckpt", strict=False, dataset='imagenet', maxpool1=False, first_conv=False, input_height=28)
	 net.freeze()

	 encoder = nn.Sequential(
   	net,
   	hd.RandomProjectionEncoder(2048, args.D)
	 )

	 classifier = hd.HDClassifier(10, args.D)

	 model = (encoder, classifier)	
	
	 
	 model_trainer = MyModelTrainer(model)
	


    aggregator = FedAVGAggregator(train_data_global, test_data_global, train_data_num,
                                  train_data_local_dict, test_data_local_dict, train_data_local_num_dict,
                                  args.client_num_per_round, device, args, model_trainer)
    
    size = args.client_num_per_round + 1
    
    
    
    server_manager = FedAVGServerManager(args,
                                         aggregator,
                                         rank=0,
                                         size=size,
                                         backend="MQTT",
                                         mqtt_host=args.mqtt_host,
                                         mqtt_port=args.mqtt_port,
                                         is_preprocessed=args.is_preprocessed)
    
    
    server_manager.run()

    # if run in debug mode, process will be single threaded by default
    #app.run(host="127.0.0.1", port=5000)
    app.run(host="0.0.0.0", port=5000)