'''
    This file contains functions for point cloud viewing and processing.
'''

import cv2 as cv
import numpy as np

def extrapolate_depth(image_data):
    '''
        Takes in value of dict returned by generator IO.read_dataset_road.next()
        and returns a matrix representing the (x,y,z) coordinate of each pixel
        in 3D space.
    '''
    left = image_data['l'].reshape(375, -1, 3)
    depth = image_data['d']
    y, x = np.meshgrid(np.arange(left.shape[0]), np.arange(left.shape[1]))
    coords = np.vstack(y, x, np.ones_like(x))
