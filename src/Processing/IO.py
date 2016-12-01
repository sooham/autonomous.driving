
import csv

import os
from os.path import join, abspath

import numpy as np
from matplotlib.pyplot import imread


################ CONSTANT FOR FILES ########################
# database sizes for road dataset
ROAD_TRAIN_SIZE              = 289
ROAD_TEST_SIZE               = 290
ROAD_GT_SIZE                 = 384

STEREO_SUBDIRS               = ['data_road', 'data_road']
DATSET_TYPES                 = ['training', 'testing']
CALIB_FOLDER                 = 'calib'




################ CREATING PATHS ############################

DIR_420 = os.environ.get('ASSIGNMENT_DIR')
if not DIR_420:
    raise IOError('CSC420 shell var not set, please set it to root folder')

DIR_420_DATA = join(abspath(DIR_420), 'data')
if not DIR_420_DATA:
    raise IOError('Dataset folder cannot be found, please place as "data" under root')


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
        return {line[0]: string_to_matrix(line[1]) for line in line_reader}




def rgb2gray(img):
    ''' Convert color image to RGB '''
    return img[...,0] * np.float32(0.299) + img[...,1] * np.float32(0.578) + img[...,2] * np.float32(0.114)


def apply_distortion(data):
    train = data['train']
    labels = data['train_y_single']

    shuffle = np.arange(train.shape[0])
    np.random.shuffle(shuffle)

    class_8 = train[labels == 8]
    class_7 = train[labels == 7]
    class_6 = train[labels == 6]
    class_4 = train[labels == 4][:200]
    class_3 = train[shuffle][labels[shuffle] == 3][:200]

    class_1 = train[shuffle][labels[shuffle] == 1][:150]
    class_2 = train[shuffle][labels[shuffle] == 2][:150]
    class_5 = train[shuffle][labels[shuffle] == 5][:150]

    for i in xrange(class_8.shape[0]):
        print(i)
        aug = distort(class_8[i], 3)
        data['train'] = np.vstack((data['train'], aug))
    data['train_y_single'] = np.hstack((data['train_y_single'], np.ones(class_8.shape[0] * 3, dtype=np.float32) * 8))

    print '.'

    for i in xrange(class_7.shape[0]):
        print(i)
        aug = distort(class_7[i], 3)
        data['train'] = np.vstack((data['train'], aug))
    data['train_y_single'] = np.hstack((data['train_y_single'], np.ones(class_7.shape[0] * 3, dtype=np.float32) * 7))

    print '.'

    for i in xrange(class_6.shape[0]):
        print(i)
        aug = distort(class_6[i], 3)
        data['train'] = np.vstack((data['train'], aug))
    data['train_y_single'] = np.hstack((data['train_y_single'], np.ones(class_6.shape[0] * 3, dtype=np.float32) * 6))

    print '.'

    for i in xrange(class_5.shape[0]):
        print(i)
        aug = distort(class_5[i], 2)
        data['train'] = np.vstack((data['train'], aug))
    data['train_y_single'] = np.hstack((data['train_y_single'], np.ones(class_5.shape[0] * 2, dtype=np.float32) * 5))

    print '.'

    for i in xrange(class_4.shape[0]):
        print(i)
        aug = distort(class_4[i], 2)
        data['train'] = np.vstack((data['train'], aug))
    data['train_y_single'] = np.hstack((data['train_y_single'], np.ones(class_4.shape[0] * 2, dtype=np.float32) * 4))

    print '.'

    for i in xrange(class_3.shape[0]):
        print(i)
        aug = distort(class_3[i], 2)
        data['train'] = np.vstack((data['train'], aug))
    data['train_y_single'] = np.hstack((data['train_y_single'], np.ones(class_3.shape[0] * 2, dtype=np.float32) * 3))

    print '.'

    for i in xrange(class_2.shape[0]):
        print(i)
        aug = distort(class_2[i], 2)
        data['train'] = np.vstack((data['train'], aug))
    data['train_y_single'] = np.hstack((data['train_y_single'], np.ones(class_2.shape[0] * 2, dtype=np.float32) * 2))

    print '.'

    for i in xrange(class_1.shape[0]):
        print(i)
        aug = distort(class_1[i], 2)
        data['train'] = np.vstack((data['train'], aug))
    data['train_y_single'] = np.hstack((data['train_y_single'], np.ones(class_1.shape[0] * 2, dtype=np.float32) * 1))

    print '.'

    assert data['train'].shape[0] == data['train_y_single'].shape[0]

    shuffle = np.arange(data['train'].shape[0])
    np.random.shuffle(shuffle)
    data['train'] = data['train'][shuffle]
    data['train_y_single'] = data['train_y_single'][shuffle]



def read(color=True):
    '''
        Reads the image training and validation dataset.
        The location of the dataset folder should be defined
        by environment variable ASSIGNMENT_LOC.

        color flag returns color dataset if True.

        Returns a dictionary with keys 'train', 'val', 'train_y_single',
        'train_y_mult'
    '''

    if color:
        process = lambda img: img
        IMAGE_DIM = 3 * 128 ** 2
    else:
        process = lambda img: rgb2gray(img)
        IMAGE_DIM = 128 ** 2

    cache = os.path.join(path, 'data_color.npz' if color else 'data.npz')
    print 'checking for ' + cache

    if os.path.isfile(cache):
        print 'loading from cache'
        return dict(np.load(cache))

    result = {}

    for subdir in [TRAIN_DIR, VAL_DIR]:
        p = os.path.join(path, subdir)
        ls_p = [item for item in os.listdir(p) if item[-4:] == '.jpg']

        result[subdir] = temp = np.empty((len(ls_p), IMAGE_DIM), dtype=np.float32)

        for i in xrange(len(ls_p)):
            temp[i] = process(imread(os.path.join(p, ls_p[i]))).ravel() / np.float32(255.0)

    # get the labels for the training set single
    with open(os.path.join(path, LABELS_SINGLE_FILE), 'rb') as fopen:
        csv_reader = csv.DictReader(fopen, delimiter=',', lineterminator='\n')
        train_y_single = np.array([int(row['Label']) for row in csv_reader])
        assert train_y_single.shape[0] == TRAIN_SIZE
        result['train_y_single'] = train_y_single

    # read the output from the TAs gist detector
    with open(os.path.join(path, GIST_FILE), 'rb') as fopen:
        csv_reader = csv.DictReader(fopen, delimiter=',', lineterminator='\n')
        gist = np.array([int(row['Prediction']) for row in csv_reader][:970])
        result['gist'] = gist


    with open(os.path.join(path, LABELS_MULT_FILE), 'rb') as fopen:
        csv_reader = csv.DictReader(
            fopen,
            restkey='values', fieldnames=['Id'], delimiter=' ', lineterminator='\n'
        )
        train_y_mult = np.array([row['values'] for row in csv_reader])
        result['train_y_mult'] = train_y_mult

    if color:
        print 'applying distortions to images'
        apply_distortion(result)
        print 'total new image db size ' + str(result['train'].shape[0])

    print 'saving to ' + cache
    np.savez(cache, **result)

    return result


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
