import numpy as npy
from PIL import Image

try:
    import cupy as np
except ImportError:
    import numpy as np

from vae.utils.functionals import initialise_weight, initialise_bias
from vae.utils.functionals import relu, sigmoid


eps = 10e-8


class Encoder(object):
    def __init__(self, input_channels, layer_size, nz, 
                batch_size=64, lr=1e-3, beta1=0.9, beta2=0.999):
        """
        """
        self.input_channels = input_channels
        self.nz = nz

        self.batch_size = batch_size
        self.layer_size = layer_size

        # Initialise encoder weight
        self.W0 = initialise_weight(self.input_channels, self.layer_size)
        self.b0 = initialise_bias(self.layer_size)

        self.W_mu = initialise_weight(self.layer_size, self.nz)
        self.b_mu = initialise_bias(self.nz)

        self.W_logvar = initialise_weight(self.layer_size, self.nz)
        self.b_logvar = initialise_bias(self.nz)

        # Adam optimiser momentum and velocity
        self.lr = lr
        self.momentum = [0.0] * 6
        self.velocity = [0.0] * 6
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
