'''
    This file contains functions for point cloud viewing and processing.
'''
from __future__ import division

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from constants import *

import numpy as np

def plot_3D(img_data):
    '''
        Takes in img_data, the value in the key value pair as returned by
        read_dataset_road().next() and plots the depth in 3d
    '''
    assert 'd' in img_data
    fig = plt.figure()
    ax = Axes3D(fig)

    depth = img_data['d']
    coord = depth[:,:,:3]

    ax.set_title('Scatter plot of image')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    z = coord[:,:,2][coord[:,:,2] != 0].ravel()
    x = coord[:,:,1][coord[:,:,2] != 0].ravel()
    y = coord[:,:,0][coord[:,:,2] != 0].ravel()
    color = depth[:,:,3:][coord[:,:,2] != 0].reshape(-1, 3)

    ax.view_init(elev=-90., azim=-90.)

    ax.scatter(
        x,
        y,
        z,
        c=color,
        marker='o',
        alpha=0.6,
        edgecolors=color,
        linewidths=0,
        depthshade=False
    )
    plt.show()
    if raw_input('do you want to save? [y/N]') == 'y':
        fname = raw_input("Enter full filename")
        savefig(join(DIR_420, 'report', 'figures', fname))
