import numpy as npy
from PIL import Image

try:
    import cupy as np
except ImportError:
    import numpy as np

from vae.utils.functionals import initialise_weight, initialise_bias
from vae.utils.functionals import MSELoss
from vae.utils.functionals import relu, sigmoid


eps = 10e-8


class Decoder(object):
    def __init__(self, input_channels, layer_size, nz, 
                batch_size=64, lr=1e-3, beta1=0.9, beta2=0.999):
        
        self.input_channels = input_channels
        self.nz = nz

        self.batch_size = batch_size
        self.layer_size = layer_size

        # Initialise decoder weight
        self.W0 = initialise_weight(self.nz, self.layer_size)
        self.b0 = initialise_bias(self.layer_size)

        self.W1 = initialise_weight(self.layer_size, self.input_channels)
        self.b1 = initialise_bias(self.input_channels)

        # Adam optimiser momentum and velocity
        self.lr = lr
        self.momentum = [0.0] * 4
        self.velocity = [0.0] * 4
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0

    def forward(self, z):
        self.z = np.reshape(z, (self.batch_size, self.nz))

        self.h0_l = self.z.dot(self.W0) + self.b0
        self.h0_a = relu(self.h0_l)

        self.h1_l = self.h0_l.dot(self.W1) + self.b1
        self.h1_a = sigmoid(self.h1_l)

        self.d_out = np.reshape(self.h1_a, (self.batch_size, self.input_channels))

        return self.d_out
