import numpy as npy
from PIL import Image

cpu_enabled = 0

try:
    import cupy as np
    cpu_enabled = 1
except ImportError:
    import numpy as np

from vae.models.vae_model import VariationalAutoEncoder
from vae.utils.dataloader import mnist_reader, img_tile
from vae.utils.functionals import MSELoss


eps = 10e-8
np.random.seed(42)


def train_mnist(vae_model, data_path, label_path,
        epochs=20, batch_size=64, save_path='./artifacts', save_freq=4):
    
    # reading training data
    trainX, _ = mnist_reader(data_path, label_path)

    np.random.shuffle(trainX)

    # set overall batch length
    batch_length = len(trainX) // batch_size
    
    # loss records tracking
    total_rec_loss = .0
    total_kl_loss = .0
    total_iter = 0

    for epoch in range(epochs):
        for b_idx in range(batch_length):
            # prepare batch data
            b_start = b_idx * batch_size
            b_end = b_start + batch_size

            # ignore batch if there are insufficient training data
            if (b_end - b_start) < batch_size:
                break

            train_batch = trainX[b_start:b_end]

            # ---------------------------
            # Forward Pass
            # ---------------------------
            out, mu, logvar = vae_model.forward(train_batch)

            # Reconstruction loss
            rec_loss = MSELoss(out, np.reshape(train_batch, (batch_size, -1)))

            # K-L Divergence loss
            kl = - 0.5 * np.sum(1 + logvar - np.power(mu, 2) - np.exp(logvar))

            loss = rec_loss + kl
            loss = loss / batch_size

            # Loss record tracking
            total_rec_loss += rec_loss / batch_size
            total_kl_loss += kl / batch_size
            total_iter += 1

            # ---------------------------
            # Backward Pass
            # ---------------------------
            # calculate gradient and upate the weights using adam
            vae_model.backward(train_batch, out)
            
        print("Epochs [{:03d}/{:03d}] | RC Loss: {:.4f} | KL Loss: {:.4f}".format(
            epoch, epochs, rec_loss / batch_size, kl / batch_size 
        ))

        if epoch % save_freq == 0:
            print("Saving image ... ")
            # save reconstructed image every save freq
            img_ = np.reshape(out, newshape=train_batch.shape)
            if cpu_enabled == 1:
                img = np.array(img_)
            img_tile(img_, save_path, epoch)



if __name__ == '__main__':
    # Configurations
    in_channel = 784 # mnist img: 28 x 28
    layer_size = 128
    nz = 32

    batch_size = 64
    lr = 1e-4

    vae_model = VariationalAutoEncoder(in_channel, layer_size, nz, 
                                        batch_size=batch_size, lr=lr,
                                        beta1=0.5, beta2=0.999)
    train_mnist(vae_model, data_path='gaussian_vae/data/train-images-idx3-ubyte',
                label_path='gaussian_vae/data/train-labels-idx1-ubyte')
