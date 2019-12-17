import numpy as npy
from PIL import Image

try:
    import cupy as np
except ImportError:
    import numpy as np


def mnist_reader(data_path='../data/train-images-idx3-ubyte', 
                label_path='../data/train-labels-idx1-ubyte'):
    """
    """
    def one_hot(label, output_dim):
        one_hot = np.zeros((len(label), output_dim))
        
        for idx in range(0,len(label)):
            one_hot[idx, label[idx]] = 1
        
        return one_hot

    # Training Data
    f_data = open(data_path)
    loaded = npy.fromfile(file=f_data, dtype=np.uint8)
    trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32) / 255 # normalise to range(0, 1) 

    # Training Label
    f_label = open(label_path)
    loaded = npy.fromfile(file=f_label, dtype=np.uint8)
    trainY = loaded[8:].reshape((60000)).astype(np.int32)

    return trainX, trainY


def img_tile(imgs, path, epoch, 
            aspect_ratio=1.0, tile_shape=None, 
            border=1, border_color=0):
    """
    """
    if imgs.ndim != 3 and imgs.ndim != 4:
        raise ValueError('imgs has wrong number of dimensions.')
    n_imgs = imgs.shape[0]

    tile_shape = None

    # Grid shape
    img_shape = npy.array(imgs.shape[1:3])
    if tile_shape is None:
        img_aspect_ratio = img_shape[1] / float(img_shape[0])
        aspect_ratio *= img_aspect_ratio
        tile_height = int(npy.ceil(npy.sqrt(n_imgs * aspect_ratio)))
        tile_width = int(npy.ceil(npy.sqrt(n_imgs / aspect_ratio)))
        grid_shape = npy.array((tile_height, tile_width))
    else:
        assert len(tile_shape) == 2
        grid_shape = npy.array(tile_shape)

    # Tile image shape
    tile_img_shape = npy.array(imgs.shape[1:])
    tile_img_shape[:2] = (img_shape[:2] + border) * grid_shape[:2] - border

    # Assemble tile image
    tile_img = npy.empty(tile_img_shape)
    tile_img[:] = border_color
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            img_idx = j + i*grid_shape[1]
            if img_idx >= n_imgs:
                # No more images - stop filling out the grid.
                break

            #-1~1 to 0~1
            img = (imgs[img_idx] + 1)/2.0# * 255.0

            yoff = (img_shape[0] + border) * i
            xoff = (img_shape[1] + border) * j
            tile_img[yoff:yoff+img_shape[0], xoff:xoff+img_shape[1], ...] = img 

    path_name = path + "/iteration_%03d"%(epoch)+".jpg"
    tile_img = tile_img.squeeze()
    img = Image.fromarray(npy.uint8(tile_img * 255) , 'L')
    img.save(path_name)

    return

