from __future__ import division

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import affine_transform, rotate, zoom

# dropout (set random pixels in image to 0)
def dropout(img):
    # takes in image and sets pixels across channels to 0
    p = np.random.uniform(0.1, 0.5)
    sel = np.random.rand(*img.shape[:2]) < p
    new = np.copy(img)
    new[sel] = 0.
    return new

# dropout_channel (set random pixels in image to 0)
def dropout_channel(img):
    p = np.random.uniform(0.1, 0.5)
    sel = np.random.rand(*img.shape) < p
    new = np.copy(img)
    new[sel] = 0.
    return new

# decrease brightness multiplicative
def brightness_adjust(img):
    return np.clip(img * np.random.rand() * 1.5, 0., 1.)

def blur(img):
    return gaussian_filter(img, np.random.uniform(0.8, 1.8))

def crop_random(img):
    h = np.random.uniform(0.7, 1.2)
    w = np.random.uniform(0.7, 1.2)
    mat = np.diag((h, w, 1.))

    dh = np.random.uniform(-0.15, 0.15) * img.shape[0]
    dw = np.random.uniform(-0.15, 0.15) * img.shape[1]
    return affine_transform(img, mat, (dh, dw, 0.))

def stretch_crop(img):
    h = np.random.uniform(1.5, 2.)
    w = np.random.uniform(1.5, 2.)
    stretched = zoom(img, [h, w] + [1.] if img.ndim == 3 else [])

    new_h, new_w = stretched.shape[:2]

    i = np.random.randint(0, new_h - img.shape[0])
    j = np.random.randint(0, new_w - img.shape[1])

    return stretched[i:i+img.shape[0], j:j+img.shape[0]]

def rotate_random(img):
    deg = np.random.uniform(-1, 1) * 90
    return rotate(img, deg, reshape=False)

def distort(img, n):
    ufunc = [dropout, dropout_channel, brightness_adjust, blur, crop_random, stretch_crop, rotate_random]
    result = np.empty((n, img.shape[0]))

    for i in xrange(n):
        sel = int(np.random.rand() * 7)
        result[i] = ufunc[sel](img.reshape(128, 128, 3)).ravel()
    return result
