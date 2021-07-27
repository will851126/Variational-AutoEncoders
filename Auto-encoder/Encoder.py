import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Below we write the Encoder class by sublcassing torch.nn.Module,
     which lets us define the __init__ method storing layers as an attribute, 
     and a forward method describing the forward pass of the network."""
    def __init__(self,lantent_dims):
        super(Encoder, self).__init__()
        self.linear1=nn.Linear(784,512)
        self.linear2=nn.Linear(512,lantent_dims)

    def forward(self, x):
        x=torch.flatten(x,start_dim=1)
        x=F.relu(self.linear1(x))

        return self.linear2(x)