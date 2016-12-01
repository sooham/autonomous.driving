'''
This library is used for calculating statistic on the
dataset and an classification results. i.e F1 score,
Recall and Precision.

TODO: Look into sklearn.metrics for scikit version of cohens kappa
      and precision accuracy curve.
'''


import matplotlib.pyplot as plt
import numpy as np

def mean_data(data, cls, normalized=True):
    '''
        Show the means for the given class cls in the training
        set in the data.
    '''
    train = data['train']
    labels = data['train_y_single']
    class_items = train[labels == cls]

    if normalized:
        norm_class_items = (class_items - train.mean(axis=0)) / train.std(axis=0)
    else:
        norm_class_items = class_items

    plt.figure(0)
    plt.axis('off')
    mean = (norm_class_items.mean(axis=0).reshape((128, 128, -1)))
    plt.imshow(mean)
    plt.show()
    raw_input('waiting...')
    plt.clf()


def var_data(data, cls, normalized=True):
    '''
        Show the variance for the given class cls in the training
        set in the data.
    '''
    train = data['train']
    labels = data['train_y_single']
    class_items = train[labels == cls]

    if normalized:
        norm_class_items = (class_items - train.mean(axis=0)) / train.std(axis=0)
    else:
        norm_class_items = class_items

    plt.figure(0)
    plt.axis('off')
    var = (norm_class_items.var(axis=0).reshape((128, 128, -1)))
    plt.imshow(var)
    plt.show()
    raw_input('waiting...')
    plt.clf()

def survey(data, num=100):
    '''
    Surveys the training dataset intraclass by choosing
    and showing num random items for each class. shows each
    image for 0.1s
    '''

    for cls in range(1, 9):   # number of classes
        class_items = (data['train'][data['train_y_single'] == cls])
        num_items = class_items.shape[0]
        for j in range(num):
            plt.figure(0)
            plt.axis('off')
            plt.imshow(
                class_items[int(np.random.rand() * num_items)].reshape(
                    (128, 128, -1)
                )
            )
            plt.pause(0.1)
            plt.close()
        raw_input('done showing class ' + str(cls))
    plt.clf()

# statistics for the classifier output
def confusion_matrix(data, classifier_output, cls):
    '''
        Returns the confusion matrix for class cls on the training data
        and given classifier output classifier_output.
    '''
    assert classifier_output.shape[0] == data['train_y_single'].shape[0]

    observed = (classifier_output == cls)
    actual = (data['train_y_single'] == cls)

    TP = np.count_nonzero(np.logical_and(observed, actual))
    TN = np.count_nonzero(np.logical_and(np.invert(observed), np.invert(actual)))
    FP = np.count_nonzero(np.logical_and(observed, np.invert(actual)))
    FN = np.count_nonzero(np.logical_and(np.invert(observed), actual))

    return np.array((TP, FP, FN, TN))

precision = lambda conf_mat: float(conf_mat[0]) / (conf_mat[0] + conf_mat[1])
recall = lambda conf_mat: float(conf_mat[0]) / (conf_mat[0] + conf_mat[3])
accuracy = lambda conf_mat: float(conf_mat[0] + conf_mat[3]) / conf_mat.sum()

def F1_score(conf_mat):
    p = precision(conf_mat)
    r = recall(conf_mat)

    return 2 * (p * r) / (p + r)
