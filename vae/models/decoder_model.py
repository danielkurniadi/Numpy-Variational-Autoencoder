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

    def optimise(self, grads):
        """
        """
        # ---------------------------
        # Optimise using Adam
        # ---------------------------
        self.t += 1
        # Calculate gradient with momentum and velocity
        for i, grad in enumerate(grads):
            self.momentum[i] = (1 - self.beta1) * grad
            self.velocity[i] = self.beta2 * self.velocity[i] + (1 - self.beta2) * np.power(grad, 2)
            m_h = self.momentum[i] / (1 - (self.beta1 ** self.t))
            v_h = self.velocity[i] /  (1 - (self.beta2 ** self.t))
            grads[i] = m_h / np.sqrt(v_h + eps)

        grad_dW0, grad_db0, grad_dW1, grad_db1 = grads

        # Update weights
        self.W0 = self.W0 - self.lr * np.sum(grad_dW0, axis=0)
        self.b0 = self.b0 - self.lr * np.sum(grad_db0, axis=0)
        self.W1 = self.W1 - self.lr * np.sum(grad_dW1, axis=0)
        self.b1 = self.b1 - self.lr * np.sum(grad_db1, axis=0)

        return

    def backward(self, x, out):
        # ----------------------------------------
        # Calculate gradients from reconstruction
        # ----------------------------------------
        y = np.reshape(x, (self.batch_size, -1))
        out = np.reshape(out, (self.batch_size, -1))

        dL = MSELoss(out, y, derivative=True)
        dSig = sigmoid(self.h1_l, derivative=True)

        dL_dSig = dL * dSig

        grad_db1 = dL_dSig
        grad_dW1 = np.matmul(np.expand_dims(self.h0_a, axis=-1), np.expand_dims(dL_dSig, axis=1))
        
        drelu0 = relu(self.h0_l, derivative=True)

        grad_db0 = grad_db1.dot(self.W1.T) * drelu0
        grad_dW0 = np.matmul(np.expand_dims(self.z, axis=-1), np.expand_dims(grad_db0, axis=1))

        # output gradient to the encoder layer
        grad_dec = grad_db0.dot(self.W0.T)

        grads = [grad_dW0, grad_db0, grad_dW1, grad_db1]

        # Optimiser Step
        self.optimise(grads)

        return grad_dec