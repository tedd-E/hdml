import torch
import torch.nn as nn
import torch_hd.hdlayers as hd


# create encoder
encoder = hd.RandomProjectionEncoder(2048, 10000)

#save to disk
torch.save(encoder.state_dict(), "/home/ersp2021/HD_Sequantial_Contianer/encoder.ckpt")
