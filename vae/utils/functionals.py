import numpy as npy

try:
    import cupy as np
except ImportError:
    import numpy as np


eps = 10e-8


# -------------------------------
# Weight Initialisation
# -------------------------------

def initialise_weight(in_channel, out_channel):
    """
    """
    W = np.random.randn(in_channel, out_channel).astype(np.float32) * np.sqrt(2.0/(in_channel))
    return W


def initialise_bias(out_channel):
    """
    """
    b = np.zeros(out_channel).astype(np.float32)
    return b


# -------------------------------
# Loss Functions
# -------------------------------

def BCELoss(x, y, derivative=False):
    """
    """
    def _BCE_loss_forward(x, y):
        loss = np.sum(- y * np.log(x + eps) + - (1 - y) * np.log((1 - x) + eps))
        return loss

    def _BCE_loss_derivative(x, y):
        dloss = -y * (1 / (x + eps))
        return dloss
    
    if derivative:
        return _BCE_loss_derivative(x, y)
    else:
        return _BCE_loss_forward(x, y)


def MSELoss(x, y, derivative=False):
    """
    """
    def _MSE_loss_forward(x, y):
        loss = (np.square(y - x)).mean()
        return loss

    def _MSE_loss_derivative(x, y):
        dloss = 2 * (x - y)
        return dloss
    
    if derivative:
        return _MSE_loss_derivative(x, y)
    else:
        return _MSE_loss_forward(x, y)


# -------------------------------
# Activation Functions
# -------------------------------

def sigmoid(x, derivative=False):
    res = 1/(1+np.exp(-x))
    if derivative:
        return res*(1-res)
    return res

def relu(x, derivative=False):
    res = x
    if derivative:
        return 1.0 * (res > 0)
    else:
        return res * (res > 0)   
    
def lrelu(x, alpha=0.01, derivative=False):
    res = x
    if derivative:
        dx = np.ones_like(res)
        dx[res < 0] = alpha
        return dx
    else:
        return np.maximum(x, x*alpha, x)

def tanh(x, derivative=False):
    res = np.tanh(x)
    if derivative:
        return 1.0 - np.tanh(x) ** 2
    return res
