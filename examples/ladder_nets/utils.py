import numpy as np
from PIL import Image

import theano
th_rng = theano.tensor.shared_randomstreams.RandomStreams(9999)

np.random.seed(9999)


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    Parameters
    ----------
    X: a 2-D ndarray or a tuple of 4 channels
        A 2-D array in which every row is a flattened image, the elements of
        which can be 2-D ndarrays or None.
    img_shape: tuple (height, width)
        The original shape of each image.
    tile_shape: tuple (nrows, ncols)
        The number of images to tile (rows, cols).
    tile_spacing: tuple, default (0, 0)
        Spacing of the tiles.
    scale_rows_to_unit_interval: bool, default True
        If True, if the values need to be scaled before being plotted to [0,1].
    output_pixel_vals: bool, default True
        If True, output should be pixel values (i.e. int8 values), otherwise
        floats.

    Returns
    -------
    array suitable for viewing as an image. (See:`Image.fromarray`.)
    """
    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    out_shape = [(ishp + tsp) * tshp - tsp
                 for ishp, tshp, tsp in
                 zip(img_shape, tile_shape, tile_spacing)]

    # if we are dealing with only one channel
    height, width = img_shape
    height_s, width_s = tile_spacing

    # generate a matrix to store the output
    dt = X.dtype
    if output_pixel_vals:
        dt = 'uint8'
    out_array = np.zeros(out_shape, dtype=dt)

    for tile_row in xrange(tile_shape[0]):
        for tile_col in xrange(tile_shape[1]):
            if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                this_x = X[tile_row * tile_shape[1] + tile_col]
                if scale_rows_to_unit_interval:
                    # if we should scale values to be between 0 and 1
                    # do this by calling the `scale_to_unit_interval`
                    # function
                    this_img = scale_to_unit_interval(
                        this_x.reshape(img_shape))
                else:
                    this_img = this_x.reshape(img_shape)

                # add the slice to the corresponding position in the
                # output array
                c = 1
                if output_pixel_vals:
                    c = 255

                tile_h = tile_row * (height + height_s)
                tile_w = tile_col * (width + width_s)

                out_array[tile_h:tile_h + height,
                          tile_w: tile_w + width] = this_img * c
    return out_array


def binarize(X, err=1e-15):
    X_mean = X.min(axis=1, keepdims=True)
    X_ptp = X.ptp(axis=1) + err
    X_norm = (X - X_mean) / X_ptp
    return th_rng.binomial(pvals=X_norm, dtype=X.dtype)


def half_linear(x):
    return 0.5 * x


def z_vals(dist, shape):
    if dist == 'Gaussian':
        return np.random.randn(*shape).astype(np.float32)
    elif dist == 'Laplacian':
        return np.random.laplace(loc=0.0, scale=np.sqrt(0.5),
                                 size=shape).astype(np.float32)


def visualize (it, images, shape=[30,30], name='samples_', p=0):
    image_data = tile_raster_images(images, img_shape=[28,28], tile_shape=shape,
                                    tile_spacing=(2,2))
    im_new = Image.fromarray(np.uint8(image_data))
    im_new.save(name+str(it)+'.png')