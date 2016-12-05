from __future__ import division
import csv

import os
from os.path import join, abspath
import math

import numpy as np
from matplotlib.pyplot import imread

from constants import *

############### HELPER FUNCTIONS ###########################

string_to_matrix = lambda s: np.array(map(np.float32, s.split())).reshape(3, -1)

def get_calib_matries(path, fname):
    '''
    Return the calibration matrices for the given path filename without extension.
    The output is a dictionary of numpy ndarrays using the original keys in
    the dataset.
    '''
    with open(join(path, fname + '.txt'), 'rb') as fopen:
        line_reader = csv.reader(fopen, delimiter=':', lineterminator='\n')
        calib = {line[0]: string_to_matrix(line[1]) for line in line_reader}
        calib['Tr_cam_to_road'] = np.vstack((calib['Tr_cam_to_road'], np.array([0., 0., 0., 1.])))
        extended_R0 = np.diag([0., 0., 0., 1.])
        extended_R0[:3, :3] = calib['R0_rect']
        calib['R0_rect'] = extended_R0
        return calib

def get_image(path, fname):
    '''
    Return the image for the given path filename without extension.
    The output is a flattened 3 channel numpy ndarray representing the image.
    '''
    # no need to divide by 255, it is already in float
    return imread(join(path, fname + '.png'))

def get_npy(path, fname):
    '''
    Return the depth array saved as .npy for the given path filename without extension.
    '''
    return np.load(join(path, fname + '.npy'))

def get_ground_images(path, fname):
    '''
    Return the ground truth images for the given path filename without extension.
    The output is a dictionary with keys 'lane' and /or 'road'.
    '''
    result = {}

    category, number = fname.split('_')

    if category == 'um':
        result['lane'] = get_image(path, category + '_lane_' + number)

    result['road'] = get_image(path, category + '_road_' + number)

    return result

#################  MAIN FUNCTION ###########################
def read_dataset_road(typ, batch_size, mode='lrdic'):
    '''
        This is a generator for the road dataset as it is too large
        to fit in memory in its entirety. Returns left right stereo pairs
        and the ground truth in batches of batch_size.

        typ: 'training' or 'testing'

        mode: any subset of the string "lrdgc" for training sets
        specifies the images per batch returned, "l" = left, "r" = right
        "g" = ground truth, "c" = calibration matrix. "d" = depth,
        "i" for disparity

        The generator will throw StopIteration Error when done.

        If typ == 'testing' set the mode 'g' is not avaliable.
    '''
    assert typ in ['testing', 'training']
    assert batch_size > 0 and type(batch_size) == int
    assert ('g' not in mode) if typ == 'testing' else True
    assert len(mode) >= 1

    flag_to_func = {
        'l': get_image,
        'r': get_image,
        'c': get_calib_matries,
        'g': get_ground_images,
        'd': get_npy,
        'i': get_npy
    }

    dir_paths = {}

    if 'r' in mode:
        dir_paths['r'] = join(DIR_420_DATA, 'data_road_right', typ, RIGHT_IMAGE_FOLDER)
    if 'l' in mode:
        dir_paths['l'] = join(DIR_420_DATA, 'data_road', typ, LEFT_IMAGE_FOLDER)
    if 'c' in mode:
        dir_paths['c'] = join(DIR_420_DATA, 'data_road', typ, CALIB_FOLDER)
    if 'g' in mode:
        dir_paths['g'] = join(DIR_420_DATA, 'data_road', typ, GROUND_IMAGE_FOLDER)
    if 'd' in mode:
        dir_paths['d'] = join(DIR_420_DATA, 'data_road', typ, DEPTH_IMAGE_FOLDER)
    if 'i' in mode:
        dir_paths['i'] = join(DIR_420_DATA, 'data_road', typ, DISP_IMAGE_FOLDER)

    listing = [fname.split('.')[0] for fname in os.listdir(
        join(DIR_420_DATA, 'data_road', typ, LEFT_IMAGE_FOLDER)
    ) if fname != '.DS_Store']

    cur = 0
    n_batches = math.ceil(len(listing) / batch_size)

    while cur < n_batches:
        result = {}
        for fname in listing[cur * batch_size: (cur+1) * batch_size]:
            current_file_info  = {}
            for flag in mode:
                current_file_info[flag] = flag_to_func[flag](dir_paths[flag], fname)

            result[fname] = current_file_info

        yield result
        cur += 1


# TODO: THIS NEEDS TO BE FIXED
def write(filename, val_predictions, hidden_predictions, include_hidden=False):
    '''
        val_predictions is a numpy.ndarray of class values for the validation set.
        hidden_predictions is a numpy.ndarray of class values for the hidden set.
        include_hidden is a parameter which takes into consideration if hidden
        validation set has been released yet.

        ASSUMES all classes in the predictions are still sorted by ID.
    '''
    if not include_hidden:
        hidden_predictions = np.zeros((2000,), dtype=np.uint8)

    with open(filename, 'w') as csvfile:
        csvfile.write('Id,Prediction\n')
        k = 0
        for i in xrange(len(val_predictions)):
            csvfile.write('%d,%d\n' % (i+1, val_predictions[i]))
            k += 1

        for j in xrange(i+1, i+1+len(hidden_predictions)):
            csvfile.write('%d,%d\n' % (j+1, hidden_predictions[j-i-1]))
            k += 1
