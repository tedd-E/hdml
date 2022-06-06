import torch
import torch.nn as nn
import torch_hd.hdlayers as hd


# create encoder
encoder = hd.RandomProjectionEncoder(2048, 10000)

#load from disk
encoder.load_state_dict(torch.load("encoder.ckpt"))
