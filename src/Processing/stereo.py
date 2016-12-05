'''
    This file contains functions for disparity and depth calculations.
'''
from __future__ import division

import os

import cv2 as cv
import numpy as np
import scipy.misc

from IO import read_dataset_road
from constants import *

###################### HELPER_FUNCTIONS ####################
def rgb2gray(img):
    ''' Convert color image to RGB '''
    return img[...,0] * np.float32(0.299) + img[...,1] * np.float32(0.578) + img[...,2] * np.float32(0.114)

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

###################### MAIN DISPARITY FUNCTIONS ######################
def cv_get_disparity(image_data, stereo):
    '''
        Takes in image_data, the value in the key value pair as returned by
        read_dataset_road().next(), and computes disparity between the left
        and right images in image_data using OpenCV stereoBM object.
    '''
    assert 'l'  in image_data and 'r' in image_data

    left = (rgb2gray(image_data['l']) * 255).astype(np.uint8)
    right = (rgb2gray(image_data['r']) * 255).astype(np.uint8)

    d = stereo.compute(left, right).astype(np.float32)

    # disparity cannot be negative, as that would mean negative depth
    d[d <= 0.] = 1e-4
    return d


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

    padded_left = np.full((left.shape[0] + 2*k, left.shape[1] + 2*k, IMG_CHANNELS), np.nan)
    padded_left[k:-k, k:-k, :] = left

    padded_right = np.full((right.shape[0] + 2*k, right.shape[1] + 2*k, IMG_CHANNELS), np.nan)
    padded_right[k:-k, k:-k, :] = right

    for y in xrange(left.shape[0]):
        print 'row ' + str(y)
        for x in xrange(1, left.shape[1]):
            # get a patch of size 2k + 1 * 2k + 1
            # with all parts outside left image as NaN
            left_patch = padded_left[y:y+2*k+1, x:x+2*k+1, :]
            # scan the row y till column x in right image
            # and get x coord with maximum NCC
            disparity_map[y,x] = x - scanline(y, x, padded_right, left_patch)

    return disparity_map

def cv_get_disparity_dataset_road(typ):
    '''
        Computes and stores the disparity of dataset typ in /disp folder
        under the images of typ using OpenCV.
    '''
    assert typ in ['training', 'testing']

    disp_path = join(DIR_420_DATA, 'data_road', typ, DISP_IMAGE_FOLDER)
    # check if folder at disp_path exists otherwise create it
    if not os.path.exists(disp_path):
        os.makedirs(disp_path)

    gen = read_dataset_road(typ, 1, 'lr')
    stereo = cv.StereoBM_create()
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=9)

    i = 0
    while True:
        try:
            print i
            image_data = gen.next()
            fname = image_data.keys()[0]
            disparity_image = cv_get_disparity(image_data[fname], stereo)
            # save it with fname
            np.save(join(disp_path, fname), disparity_image)
            i += 1
        except StopIteration:
            break

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
            disparity_image = get_disparity(image_data[fname])
            # save it with fname
            scipy.misc.imsave(join(disp_path, fname + '.png'), disparity_image)
            i += 1
        except StopIteration:
            break

###################### MAIN DEPTH FUNCTIONS ################

def get_depth(image_data):
    '''
        Takes in image_data, the value in the key value pair as returned by
        read_dataset_road().next(), and computes depth from the previously
        found disparity.

        Depth is stored in a .npy file as a [IMAGE_HEIGHT, IMAGE_WIDTH, ]
    '''
    assert 'l' in image_data and 'i' in image_data and 'c' in image_data
    P2 = image_data['c']['P2']
    P3 = image_data['c']['P3']
    disp = image_data['i']

    f = P2[0,0]
    T = P2[0,3] - P3[0,3]

    principal = P2[:2, 2]

    mat = np.diag((1, 1, f))
    mat[:2, 2] = -principal[::-1]
    mat = mat.T

    x, y = np.meshgrid(np.arange(disp.shape[1]), np.arange(disp.shape[0]))
    grid = np.dstack((y, x, np.ones_like(x)))

    depth = np.empty_like(grid, dtype=np.float32)

    for y in xrange(disp.shape[0]):
        for x in xrange(disp.shape[1]):
            depth[y, x] = np.dot(grid[y, x], mat) / disp[y,x]

    depth *= T

    # This is to set all invalid points to depth 0
    d =depth[:,:,2]
    max_d = d.max()
    d[d == max_d] = 0.

    depth = np.dstack((depth, image_data['l']))
    return depth

def get_depth_dataset_road(typ):
    '''
        Computes and stores the depth of dataset typ in /depth folder
        under the images of typ.
    '''
    assert typ in ['training', 'testing']

    depth_path = join(DIR_420_DATA, 'data_road', typ, DEPTH_IMAGE_FOLDER)
    # check if folder at disp_path exists otherwise create it
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    gen = read_dataset_road(typ, 1, 'lic')

    i = 0
    while True:
        try:
            print i
            image_data = gen.next()
            fname = image_data.keys()[0]
            depth_image = get_depth(image_data[fname])
            # save it with fname
            np.save(join(depth_path, fname), depth_image)
            i += 1
        except StopIteration:
            break