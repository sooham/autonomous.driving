'''
    This file contains functions for disparity and depth calculations.
'''
from __future__ import division

import os

import numpy as np
import scipy.misc

from IO import read_dataset_road
from constants import *

###################### HELPER_FUNCTIONS ####################
def NCC(patch1, patch2):
    '''
        Returns the Normalized Cross Correlation between patch1
        and patch2, with no zero padding.
    '''
    return np.correlate(patch1, patch2) / np.sqrt(np.sum(patch1 ** 2) * np.sum(patch2 ** 2))

def scanline(row, col, right_image, left_patch):
    '''
        Gets patches along the row image[row, :col] in right_image and
        does NCC comparison to left_patch. Returns the x-value along
        the row with the greatest NCC.
    '''
    k = left_patch.shape[0] // 2
    max_x = 0
    max_ncc = 0
    for x in xrange(col):
        right_patch = right_image[row:row+2*k+1, x:x+2*k+1, :]
        common = np.invert(np.logical_or(np.isnan(right_patch), np.isnan(left_patch)))
        ncc = NCC(left_patch[common], right_patch[common])
        if ncc > max_ncc:
            max_ncc = ncc
            max_x = x
    return x

###################### MAIN FUNCTIONS ######################
def get_disparity(image_data, k=2):
    '''
        Takes in image_data, the value in the key value pair as returned by
        read_dataset_road().next(), and computes disparity between the left
        and right images in image_data with patch_size 2*k+1.
    '''
    assert 'l'  in image_data and 'r' in image_data
    left = image_data['l']
    right = image_data['r']
    # the left and right images are rectified
    # hence for every point in the left image we only need to scan
    # the horizontal line
    disparity_map = np.empty_like(left)

    # the 0th column of the disparity map will be infinity
    # as scanline() will go over no elements, i.e there are no indices to left of 0
    disparity_map[:, 0] = np.inf

    padded_left = np.full((IMG_HEIGHT + 2*k, IMG_WIDTH + 2*k, IMG_CHANNELS), np.nan)
    padded_left[k:-k, k:-k, :] = left

    padded_right = np.full((IMG_HEIGHT + 2*k, IMG_WIDTH + 2*k, IMG_CHANNELS), np.nan)
    padded_right[k:-k, k:-k, :] = right

    for y in xrange(IMG_HEIGHT):
        print 'row ' + str(y)
        for x in xrange(1, IMG_WIDTH):
            # get a patch of size 2k + 1 * 2k + 1
            # with all parts outside left image as NaN
            left_patch = padded_left[y:y+2*k+1, x:x+2*k+1, :]
            # scan the row y till column x in right image
            # and get x coord with maximum NCC
            disparity_map[y,x] = x - scanline(y, x, padded_right, left_patch)

    return disparity_map

def get_disparity_dataset_road(typ):
    '''
        Computes and stores the disparity of dataset typ in /disparity folder
        under the images of typ
    '''
    assert typ in ['training', 'testing']

    disp_path = join(DIR_420_DATA, 'data_road', typ, DISP_IMAGE_FOLDER)
    # check if folder at disp_path exists otherwise create it
    if not os.path.exists(disp_path):
        os.makedirs(disp_path)

    gen = read_dataset_road(typ, 1, 'lr')

    i = 0
    while True:
        try:
            print i
            image_data = gen.next()
            fname = image_data.keys()[0]
            disparity_image = get_disparity(image_data[fname]).astype(np.uint8)
            # save it with fname
            scipy.misc.imsave(join(disp_path, fname + '.png'), disparity_image)
        except StopIteration:
            break
