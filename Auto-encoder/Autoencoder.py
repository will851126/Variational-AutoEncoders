import torch
import torch.nn as nn
import torch.nn.functional as F
from Encoder import Encoder
from Decoder import Decoder

class Autoencoder(nn.Module):
    def __init__(self,lantent_dims):
        super(Autoencoder,self).__init__()
        self.encoder=Encoder(lantent_dims)
        self.decoder=Decoder(lantent_dims)

    def forward(self, x):
        z=self.encoder(x)
        return self.decoder(z)
   
    def train(autoencoder, data, epochs=20):
        opt = torch.optim.Adam(autoencoder.parameters())
        for x, y in data:
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()



