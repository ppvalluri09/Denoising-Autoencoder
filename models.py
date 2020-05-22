import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, latent_dim)
                )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

class Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        self.net = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 784)
                )

    def forward(self, x):
        out = self.net(x)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(in_dim, latent_dim)
        self.decoder = Decoder(latent_dim, in_dim)

    def forward(self, x):
        encoder_output = self.encoder(x.view(x.size(0), -1))
        decoder_output = self.decoder(encoder_output)

        return decoder_output

class AutoEncoderConv(nn.Module):
    def __init__(self, in_channels):
        super(AutoEncoderConv, self).__init__()

        self.encode = nn.Sequential(
                    # mx1x28x28
                    nn.Conv2d(in_channels, 16, 4, 2, 1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    # mx16x14x14
                    nn.Conv2d(16, 32, 4, 2, 0),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    # mx32x5x5
                    nn.Conv2d(32, 64, 4, 1, 0)
                )

        self.decode = nn.Sequential(
                    # mx64x2x2
                    nn.ConvTranspose2d(64, 32, 4, 1, 0),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    # mx64x5x5
                    nn.ConvTranspose2d(32, 16, 4, 2, 0),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    # mx16x14x14
                    nn.ConvTranspose2d(16, in_channels, 4, 2, 1)
                )

    def forward(self, x):
        encoder_out = self.encode(x.float())
        decoder_out = self.decode(encoder_out)

        return decoder_out
