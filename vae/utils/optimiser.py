import numpy as npy

try:
    import cupy as np
except ImportError:
    import numpy as np


eps = 10e-8

# -------------------------------
# Base Optimiser
# -------------------------------

class BaseOptim(object):
    def __init__(self, params, lr, *args, **kwargs):
        pass

    def step(self, grads):
        pass


# -------------------------------
# Adam Optimiser
# -------------------------------

class AdamOptim(object):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0):
        assert len(betas) == 2
        assert abs(weight_decay) < 1
        
        self.lr = lr
        self.weight_decay = abs(weight_decay)

        self.beta1 = betas[0]
        self.beta2 = betas[1]

        self.params = params
        self.momentum = [0.0] * len(self.params)
        self.velocity = [0.0] * len(self.params)
        self.t = 0

    def step(self, grads):
        assert len(grads) == len(self.params)

        self.t += 1

        # Calculate gradient with momentum and velocity
        for i, grad in enumerate(grads):
            self.momentum[i] = self.beta1 * self.momentum[i] + (1 - self.beta1) * grad
            self.velocity[i] = self.beta2 * self.velocity[i] + (1 - self.beta2) * np.power(grad, 2)

            m_h = self.momentum[i] / (1 - (self.beta1 ** self.t))
            v_h = self.velocity[i] / (1 - (self.beta2 ** self.t))

            grads[i] = m_h / np.sqrt(v_h + eps)

        # Update weights
        for idx in range(len(self.params)):
            np.add(self.params[idx], -self.lr * (np.sum(grads[idx], axis=0)),
                    out=self.params[idx])

        return
