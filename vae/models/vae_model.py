import numpy as npy
from PIL import Image

try:
    import cupy as np
except ImportError:
    import numpy as np

from .encoder_model import Encoder
from .decoder_model import Decoder

eps = 10e-8


class VariationalAutoEncoder(object):

    def __init__(self, input_channels, layer_size, nz, 
                batch_size=64, lr=1e-3, beta1=0.9, beta2=0.999):
        
        self.input_channels = input_channels
        self.nz = nz

        self.batch_size = batch_size
        self.layer_size = layer_size

        # Construct encoder module
        self.encoder = Encoder(input_channels, layer_size, nz, 
            batch_size=batch_size, lr=lr, beta1=beta1, beta2=beta2)
        
        # Construct decoder module
        self.decoder = Decoder(input_channels, layer_size, nz,
            batch_size=batch_size, lr=lr, beta1=beta1, beta2=beta2)
        
    def forward(self, x):
        """
        """
        x = x.reshape((self.batch_size, -1))

        # Feed forward encoder - decoder
        sample_z, mu, logvar = self.encoder.forward(x)
        out = self.decoder.forward(sample_z)

        return out, mu, logvar

    def backward(self, x, out):
        """
        """
        grad_dec = self.decoder.backward(x, out)
        self.encoder.backward(x, grad_dec)


