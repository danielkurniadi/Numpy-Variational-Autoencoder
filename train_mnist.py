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
np.seterr(all='raise')


def train_mnist(vae_model, data_path, label_path,
                epochs=20, batch_size=64, print_freq=100,
                save_path='./artifacts', save_freq=2):
    """
    """

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
        for b_idx in range(batch_length//2):
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
            total_rec_loss += rec_loss
            total_kl_loss += kl / batch_size
            total_iter += 1

            # ---------------------------
            # Backward Pass
            # ---------------------------
            # calculate gradient and upate the weights using adam
            vae_model.backward(train_batch, out)

            # Logging
            if (b_idx+1) % print_freq == 0:
                print("Epochs [{:03d}/{:03d}] Iter[{:03d}/{:03d}] | RC Loss: {:.4f} | KL Loss: {:.4f}".format(
                    epoch, epochs, (b_idx+1) * batch_size, batch_length * batch_size, 
                    rec_loss, kl / batch_size
                ))

        if (b_idx+1) % save_freq == 0:
            print("Saving image ... ")
            # save 16 reconstructed images every save_freq
            imgs_ = np.reshape(out, newshape=train_batch.shape)
            if cpu_enabled == 1:
                imgs = np.array(imgs_)
            img_tile(imgs_, save_path, epoch)


if __name__ == '__main__':
    # Configurations
    in_channel = 784 # mnist img: 28 x 28
    layer_size = 128
    nz = 32

    batch_size = 64
    lr = 1e-4
    beta1 = 0.9 
    beta2 = 0.999
    weight_decay = 0.1

    optimiser_kwargs = dict(lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)

    # create model
    vae_model = VariationalAutoEncoder(in_channel, layer_size, nz, 
                                        batch_size=batch_size, **optimiser_kwargs)
    # run training session
    train_mnist(vae_model, 
                data_path='./data/train-images-idx3-ubyte',
                label_path='./data/train-labels-idx1-ubyte',
                batch_size=batch_size, print_freq=100, save_freq=1)
