'''
    This file defines commonly used constants across all programs.
'''
import os
from os.path import abspath, join

ROAD_TRAIN_SIZE              = 289
ROAD_TEST_SIZE               = 290

NUM_UM_TRAIN                 = 95
NUM_UMM_TRAIN                = 96
NUM_UU_TRAIN                 = 98

NUM_UM_TEST                  = 96
NUM_UMM_TEST                 = 94
NUM_UU_TEST                  = 100

IMG_HEIGHT                   = 375
IMG_WIDTH                    = 1242
IMG_CHANNELS                 = 3

STEREO_SUBDIRS               = ['data_road', 'data_road']
CALIB_FOLDER                 = 'calib'
LEFT_IMAGE_FOLDER            = 'image_2'
GROUND_IMAGE_FOLDER          = 'gt_image_2'
DEPTH_IMAGE_FOLDER           = 'depth'
DISP_IMAGE_FOLDER            = 'disp'
RIGHT_IMAGE_FOLDER           = 'image_3'

################ CREATING PATHS ############################
DIR_420 = os.environ.get('CSC420')
if not DIR_420:
    raise IOError('CSC420 shell var not set, please set it to root folder')

DIR_420_DATA = join(abspath(DIR_420), 'data')
if not DIR_420_DATA:
    raise IOError('Dataset folder cannot be found, please place as "data" under root')
