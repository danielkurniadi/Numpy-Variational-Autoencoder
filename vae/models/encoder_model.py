import numpy as npy
from PIL import Image

try:
    import cupy as np
except ImportError:
    import numpy as np

from vae.utils.functionals import initialise_weight, initialise_bias
from vae.utils.functionals import relu, sigmoid
from vae.utils.optimiser import AdamOptim

eps = 10e-8


class Encoder(object):
    def __init__(self, input_channels, layer_size, nz, 
                batch_size=64, **optimiser_kwargs):
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

        params = [self.W0, self.b0, self.W_mu, self.b_mu, self.W_logvar, self.b_logvar]

        # Adam optimiser momentum and velocity
        self.optimiser = AdamOptim(params, **optimiser_kwargs)

    def forward(self, x):
        """
        """
        self.e_input = x.reshape((self.batch_size, -1))
        
        # Dimension check on input
        assert self.e_input.shape == (self.batch_size, self.input_channels)

        self.h0_l = self.e_input.dot(self.W0) + self.b0
        self.h0_a = relu(self.h0_l)

        self.logvar = self.h0_a.dot(self.W_logvar) + self.b_logvar
        self.mu = self.h0_a.dot(self.W_mu) + self.b_mu

        self.rand_sample = np.random.standard_normal(size=(self.batch_size, self.nz))
        self.sample_z = self.mu + np.exp(self.logvar * .5) * self.rand_sample

        return self.sample_z, self.mu, self.logvar

    def backward(self, x, grad_dec):
        """
        """
        # ----------------------------------------
        # Calculate gradients from reconstruction
        # ----------------------------------------
        y = np.reshape(x, (self.batch_size, -1))

        db_mu = grad_dec 
        dW_mu = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(grad_dec, axis=1))

        db_logvar = grad_dec * np.exp(self.logvar * .5) * .5 * self.rand_sample
        dW_logvar = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(db_logvar, axis=1))

        drelu = relu(self.h0_l, derivative=True)

        db0 = drelu * (db_mu.dot(self.W_mu.T) + db_logvar.dot(self.W_logvar.T))
        dW0 = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(db0, axis=1))

        # ----------------------------------------
        # Calculate gradients from K-L
        # ----------------------------------------
        # logvar terms
        dKL_b_logvar = .5 * (np.exp(self.logvar) - 1)
        dKL_W_logvar = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(dKL_b_logvar, axis=1))

        # mu terms
        dKL_b_mu = .5 * 2 * self.mu
        dKL_W_mu = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(dKL_b_mu, axis=1))

        dKL_b0 = drelu * (dKL_b_logvar.dot(self.W_logvar.T) + dKL_b_mu.dot(self.W_mu.T))
        dKL_W0 = np.matmul(np.expand_dims(y, axis=-1), np.expand_dims(dKL_b0, axis=1))

        # Combine gradients for encoder from recon and KL
        grad_b_logvar = dKL_b_logvar + db_logvar
        grad_W_logvar = dKL_W_logvar + dW_logvar
        grad_b_mu = dKL_b_mu + db_mu
        grad_W_mu = dKL_W_mu + dW_mu
        grad_b0 = dKL_b0 + db0
        grad_W0 = dKL_W0 + dW0

        grads = [grad_W0, grad_b0, grad_W_mu, grad_b_mu, grad_W_logvar, grad_b_logvar]

        # Optimise step
        self.optimiser.step(grads)

        return
