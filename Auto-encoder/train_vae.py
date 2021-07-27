import torch
import torch.nn as nn
import torch.nn.functional as F
from Autoencoder import Autoencoder
from Encoder import Encoder
from Decoder import Decoder
from VariationalEncoder import VariationalEncoder
from VariationalEncoder import VariationalAutoencoder
import torchvision



lantent_dims = 2

vae = VariationalAutoencoder(lantent_dims)

data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
               transform=torchvision.transforms.ToTensor(),
               download=True),
               batch_size=128,
               shuffle=True)

vae = train(vae, data)
