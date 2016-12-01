############################# IMPORTS #######################################
from __future__ import division
from __future__ import print_function

import sys
import os
import math
import numpy as np
import tensorflow as tf

############################# CONSTANTS ########################################
ASSIGNMENT_DIR = os.path.abspath((os.environ.get('ASSIGNMENT_DIR')))
SAVE_DIR = os.path.join(ASSIGNMENT_DIR, 'report/results/save/')

sys.path.insert(0,
    os.path.join(ASSIGNMENT_DIR, 'src')
)

from Processing.IO import read, write


############################# load the training data ###########################
color = True
data = read(color=color)

train = data['train'][350:]
valid = data['train'][:350]
test = data['val']

l_train = data['train_y_single'][350:] - 1
l_valid = data['train_y_single'][:350] - 1

n_classes = 8
n_data = train.shape[0]
n_valid = valid.shape[0]

# create one-hot representation of data
labels_train = np.zeros((n_data, n_classes), dtype=np.float32)
labels_train[np.arange(n_data), l_train] = 1.0

labels_valid = np.zeros((n_valid, n_classes), dtype=np.float32)
labels_valid[np.arange(n_valid), l_valid] = 1.0
# computations
n_channel = 3 if color else 1

############################# hyper-parameters #################################
learning_rate = 0.00015
n_epoches = 2000
batch_size = 150 # 500

# Network Parameters
n_input = 128 * 128 * n_channel
dropout = 0.85
momentum = 0.002          # used if momentum implemented


# computations
n_batches = int(math.ceil(n_data / batch_size))

############################ helper functions #################################
def conv2d(tensor, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    tensor_conv = tf.nn.conv2d(tensor, W, strides=[1, strides, strides, 1], padding='SAME')
    tensor_add_bias = tf.nn.bias_add(tensor_conv, b)
    return tf.nn.relu(tensor_add_bias)


def maxpool2d(tensor, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


########################### covnet model ######################################
def conv_net(tensor, weights, biases, keep_p):
    # Reshape input picture
    tensor_r = tf.reshape(tensor, shape=[-1, 128, 128, n_channel])
    #tf.image_summary('tensor_input', tensor_r, max_images=50)

    # Convolution Layer
    conv1_conv = conv2d(tensor_r, weights['wc1'], biases['bc1'])
    #tf.histogram_summary('conv1 layer conv', conv1_conv)

    conv1_pool = maxpool2d(conv1_conv, k=2)

    conv2_conv = conv2d(conv1_pool, weights['wc2'], biases['bc2'])
    #tf.histogram_summary('conv2 layer conv', conv2_conv)

    # Max Pooling (down-sampling)
    conv2_pool = maxpool2d(conv2_conv, k=2)

    # Convolution Layer
    conv3_conv = conv2d(conv2_pool, weights['wc3'], biases['bc3'])
    #tf.histogram_summary('conv3 layer conv', conv3_conv)

    conv3_pool = maxpool2d(conv3_conv, k=2)

    # Convolution Layer
    conv4_conv = conv2d(conv3_pool, weights['wc4'], biases['bc4'])
    #tf.histogram_summary('conv4 layer conv', conv4_conv)

    conv4_pool = maxpool2d(conv4_conv, k=2)

    conv5_conv = conv2d(conv4_pool, weights['wc5'], biases['bc5'])
    #tf.histogram_summary('conv5 layer conv', conv5_conv)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1_reshape = tf.reshape(conv5_conv, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1_preactivate = tf.add(tf.matmul(fc1_reshape, weights['wd1']), biases['bd1'])
    #tf.image_summary('fully connected preactivate', tf.reshape(fc1_preactivate, [1, batch_size, -1, n_channel]))

    fc1_relu = tf.nn.relu(fc1_preactivate)
    #tf.image_summary('fully connected RELU', tf.reshape(fc1_relu, [1, batch_size, -1, n_channel]))
    # Apply Dropout
    fc1_dropout = tf.nn.dropout(fc1_relu, keep_p)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1_dropout, weights['out']), biases['out_b'])
    return out

########################## tf Graph inputs ####################################
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

########################## weights and biases #################################
weights = {
    # 5x5 conv, 1 input, 40 outputs
    'wc1': tf.Variable(tf.truncated_normal([7, 7, n_channel, 56], stddev=0.1)),
    # 5x5 conv, 40 inputs, 40 outputs
    'wc2': tf.Variable(tf.truncated_normal([7, 7, 56, 56], stddev=0.1)),
    # 5x5 conv, 40 inputs, 64 outputs
    'wc3': tf.Variable(tf.truncated_normal([7, 7, 56, 64], stddev=0.1)),
    # 5x5 conv, 64 inputs, 64 outputs
    'wc4': tf.Variable(tf.truncated_normal([7, 7, 64, 112], stddev=0.1)),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wc5': tf.Variable(tf.truncated_normal([7, 7, 112, 256], stddev=0.1)),

    'wd1': tf.Variable(tf.truncated_normal([8*8*256*n_channel, 3069], stddev=0.1)),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([3069, n_classes], stddev=0.1))
}

biases = {
    'bc1': tf.Variable(tf.constant(0.1, shape=[56])),
    'bc2': tf.Variable(tf.constant(0.1, shape=[56])),
    'bc3': tf.Variable(tf.constant(0.1, shape=[64])),
    'bc4': tf.Variable(tf.constant(0.1, shape=[112])),
    'bc5': tf.Variable(tf.constant(0.1, shape=[256])),
    'bd1': tf.Variable(tf.constant(0.1, shape=[3069])),
    'out_b': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

# Construct model
logits = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)

# Evaluate model
predictions = tf.argmax(logits, 1)

correct_pred = tf.cast(tf.equal(predictions, tf.argmax(y, 1)), tf.float32)

accuracy = tf.reduce_mean(correct_pred)

accuracy_valid = tf.reduce_mean(correct_pred)

# Initializing the variables
init = tf.initialize_all_variables()

########################### logging statistics #################################
# create a Saver to save the variables
saver = tf.train.Saver()

with tf.name_scope('conv1'):
    tf.scalar_summary('wc1/mean', tf.reduce_mean(weights['wc1']))
    tf.scalar_summary('wc1/min', tf.reduce_min(weights['wc1']))
    tf.scalar_summary('wc1/max', tf.reduce_max(weights['wc1']))
    #tf.histogram_summary('conv1/wc1', weights['wc1'])

    tf.scalar_summary('bc1/mean', tf.reduce_mean(biases['bc1']))
    tf.scalar_summary('bc1/min', tf.reduce_min(biases['bc1']))
    tf.scalar_summary('bc1/max', tf.reduce_max(biases['bc1']))
    #tf.histogram_summary('conv1/bc1', biases['bc1'])

with tf.name_scope('conv2'):
    tf.scalar_summary('wc2/mean', tf.reduce_mean(weights['wc2']))
    tf.scalar_summary('wc2/min', tf.reduce_min(weights['wc2']))
    tf.scalar_summary('wc2/max', tf.reduce_max(weights['wc2']))
    #tf.histogram_summary('conv2/wc2', weights['wc2'])

    tf.scalar_summary('bc2/mean', tf.reduce_mean(biases['bc2']))
    tf.scalar_summary('bc2/min', tf.reduce_min(biases['bc2']))
    tf.scalar_summary('bc2/max', tf.reduce_max(biases['bc2']))
    #tf.histogram_summary('conv2/bc2', biases['bc2'])

with tf.name_scope('conv3'):
    tf.scalar_summary('wc3/mean', tf.reduce_mean(weights['wc3']))
    tf.scalar_summary('wc3/min', tf.reduce_min(weights['wc3']))
    tf.scalar_summary('wc3/max', tf.reduce_max(weights['wc3']))
    #tf.histogram_summary('conv3/wc3', weights['wc3'])

    tf.scalar_summary('bc3/mean', tf.reduce_mean(biases['bc3']))
    tf.scalar_summary('bc3/min', tf.reduce_min(biases['bc3']))
    tf.scalar_summary('bc3/max', tf.reduce_max(biases['bc3']))
    #tf.histogram_summary('conv3/bc3', biases['bc3'])


with tf.name_scope('conv4'):
    tf.scalar_summary('wc4/mean', tf.reduce_mean(weights['wc4']))
    tf.scalar_summary('wc4/min', tf.reduce_min(weights['wc4']))
    tf.scalar_summary('wc4/max', tf.reduce_max(weights['wc4']))
    #tf.histogram_summary('conv4/wc4', weights['wc4'])

    tf.scalar_summary('bc4/mean', tf.reduce_mean(biases['bc4']))
    tf.scalar_summary('bc4/min', tf.reduce_min(biases['bc4']))
    tf.scalar_summary('bc4/max', tf.reduce_max(biases['bc4']))
    #tf.histogram_summary('conv4/bc4', biases['bc4'])

with tf.name_scope('conv5'):
    tf.scalar_summary('wc5/mean', tf.reduce_mean(weights['wc5']))
    tf.scalar_summary('wc5/min', tf.reduce_min(weights['wc5']))
    tf.scalar_summary('wc5/max', tf.reduce_max(weights['wc5']))
    #tf.histogram_summary('conv4/wc4', weights['wc4'])

    tf.scalar_summary('bc5/mean', tf.reduce_mean(biases['bc5']))
    tf.scalar_summary('bc5/min', tf.reduce_min(biases['bc5']))
    tf.scalar_summary('bc5/max', tf.reduce_max(biases['bc5']))
    #tf.histogram_summary('conv4/bc4', biases['bc4'])

with tf.name_scope('full1'):
    tf.scalar_summary('wd1/mean', tf.reduce_mean(weights['wd1']))
    tf.scalar_summary('wd1/min', tf.reduce_min(weights['wd1']))
    tf.scalar_summary('wd1/max', tf.reduce_max(weights['wd1']))
    #tf.histogram_summary('full/wd1', weights['wd1'])

    tf.scalar_summary('bd1/mean', tf.reduce_mean(biases['bd1']))
    tf.scalar_summary('bd1/min', tf.reduce_min(biases['bd1']))
    tf.scalar_summary('bd1/max', tf.reduce_max(biases['bd1']))
    #tf.histogram_summary('full/bd1', biases['bd1'])

with tf.name_scope('out'):
    tf.scalar_summary('out/mean', tf.reduce_mean(weights['out']))
    tf.scalar_summary('out/min', tf.reduce_min(weights['out']))
    tf.scalar_summary('out/max', tf.reduce_max(weights['out']))
    #tf.histogram_summary('out/out_image', weights['out'])
    #tf.histogram_summary('out/out', weights['out'])

    tf.scalar_summary('out_b/mean', tf.reduce_mean(biases['out_b']))
    tf.scalar_summary('out_b/min', tf.reduce_min(biases['out_b']))
    tf.scalar_summary('out_b/max', tf.reduce_max(biases['out_b']))
    #tf.histogram_summary('out/out_b', biases['out_b'])

tf.image_summary('CNN logits', tf.reshape(logits, [1, batch_size, n_classes, 1]))
tf.scalar_summary('accuracy', accuracy)
tf.scalar_summary('accuracy validation', accuracy_valid)
# TODO: Implement F1 score for x and y tensor
tf.scalar_summary('(cost) cross entropy', cost)
#tf.histogram_summary('num correct', correct_pred)
#tf.histogram_summary('predictions', predictions)

# create summary writer for training
summaries = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(
    os.path.join(ASSIGNMENT_DIR, 'report/results/tmp'),
    graph=optimizer.graph
    )

############################# event loop ######################################
with tf.Session() as sess:
    sess.run(init)

    # ask if to restore variables?
    if raw_input('Do you want to restore weights from a previous Session? [y/N] ') == 'y':
        fname = raw_input('Enter the filename \n')
        saver.restore(sess, os.path.join(SAVE_DIR, fname))
        print('Model restored')


    epoch = 0
    indices = np.arange(train.shape[0])
    # Keep training until reach max iterations
    while epoch < n_epoches:
        np.random.shuffle(indices)
        shuffled_labels, shuffled_train = labels_train[indices], train[indices]

        for b in xrange(n_batches):
            batch_y = shuffled_labels[b*batch_size:(b+1)*batch_size]
            batch_x = shuffled_train[b*batch_size:(b+1)*batch_size]

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})

        # DONE WITH EPOCH
        # Calculate batch loss and accuracy

        acc_valid = sess.run(accuracy_valid, feed_dict={x: valid, y: labels_valid, keep_prob: 1.})

        loss, acc, pred, summ  = sess.run([
            cost, accuracy, predictions, summaries
            ], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})

        writer.add_summary(summ, global_step=epoch + 1)
        print(pred.min(), pred.max())
        print("Epoch: " + str(epoch + 1) + ", Minibatch Loss= " + \
              "{:.5f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc) + ", Validation Accuracy=" + "{:.5f}".format(acc_valid))

        epoch += 1

    print("Optimization Finished!")

    # close the writer
    writer.close()

    # run on the valiation set and write to file
    test_cls = sess.run(predictions, feed_dict={x: test, keep_prob: 1.})
    print(test_cls.min(), test_cls.max())
    test_cls += 1

    filename = os.path.join(ASSIGNMENT_DIR, 'report/results/output.csv')
    write(filename, test_cls, None, include_hidden=False)

    # save the model variables
    if raw_input('Do you want to save the model weights? [y/N]') == 'y':
        fname = raw_input('Enter the filename \n')
        save_path = saver.save(sess, os.path.join(SAVE_DIR, fname))
        print("Model saved in file: %s" % save_path)
