import copy
import logging
import time

import numpy as np
import wandb

import torch

from .utils import transform_list_to_tensor


class FedAVGAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer):
        self.classifier = model_trainer

        self.train_global = train_global
        self.test_global = test_global
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.args = args
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.flag_client_model_uploaded_dict = dict()
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.classifier.class_hvs.clone().to(self.device)
    
    def set_global_model_params(self, model_parameters):
        self.classifier.class_hvs = nn.Parameter(model_parameters, requires_grad=False)

    def add_local_trained_result(self, index, model_params, sample_num):
        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        self.flag_client_model_uploaded_dict[index] = True

    def check_whether_all_receive(self):
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    def aggregate(self):
        num = len(self.model_dict)
        updated_params = self.classifier.class_hvs.clone().to(self.device)
        for model in self.model_dict:
            updated_params += model.class_hvs
        updated_params /= num
        set_model_params(updated_params)
        return updated_params

    def test_on_server_for_all_clients(self):
        accuracy = self.trainer.test(self.test_global, self.device, self.args)
