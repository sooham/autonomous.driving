'''
Simple implementation of mixture of gaussian in tensorflow
'''

from __future__ import print_function
from __future__ import division

import sys
import os

ASSIGNMENT_DIR = os.getenv('ASSIGNMENT_DIR')
sys.path.insert(0,
    os.path.join(ASSIGNMENT_DIR, 'src')
)

from Processing.IO import read, write

data = read(color=False)


# hyperparameters
K = 8               # number of clusters
iter = 10           # iterations of EM algorithm
kmeans_iters = 5    # number of Kmeans iters
minVary = 0.01      # lower bound on the variance to prevent singularities

##########################################################################
#                            KMEANS
##########################################################################

def distmat(p, q):
  """Computes pair-wise L2-distance between rows of p and q."""
  pmag = np.sum(p**2, 1)
  qmag = np.sum(q**2, 1)
  total = pmag + np.reshape(qmag, [-1, 1])

  dist = total - 2 * np.dot(q, np.transpose(p))
  return np.sqrt(dist)


def KMeans(x, n_clusters, iters):
    '''
        Cluster tensors into n_clusters using K-Means

        tensor: data tensor, each row is a data point
        n_clusters: number of clusters to K means
        iters: number of Kmeans iterations
    '''

    # get random cluster means from the tensor
    rand = np.arange(x.shape[0])
    np.random.shuffle(rand)

    cluster_centers = x[rand[:n_clusters]]

    # assign each class to closest cluster center
    for ii in xrange(iters):
        print('Kmeans iteration = %04d' % (ii + 1))
        # cluster_centers is [K, dim] shape tensor
        # need to compute the distance matrix between
        dist_mat = distmat(cluster_centers, x)
        assigned_class = np.argmin(dist_mat, 1)
        for k in xrange(n_clusters):
            cluster_centers[k] = np.mean(x[(assigned_class == k).nonzero()[0]], 0)
    return cluster_centers


##########################################################################
#                            MOG
##########################################################################

randConst = tf.placeholder(tf.float32)

def mixture_gaussians(x, n_clusters, iters):
    """
    Fits a Mixture of n_clusters Diagonal Gaussians on x.

    Inputs:
      x: data with one data vector in each row.
      n_clusters: Number of Gaussians.
      iters: Number of EM iterations.
      randConst: scalar to control the initial mixing coefficients
      minVary: minimum variance of each Gaussian.

    Returns:
      p: probabilities of clusters (or mixing coefficients).
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.
      logLikelihood: log-likelihood of data after every iteration.
    """
    T, N = tf.shape(x)

    # Initialize the parameters
    # IMPORTANT: randConstant used here to init mixing params

    # p = mixing coefficients
    p = randConst + tf.Variable(tf.random_normal([1, n_clusters]))
    p /= tf.sum(p)
    # mean across features
    mn = np.mean(x, 0).reshape(-1, 1)
    vr = np.var(x, 0).reshape(-1, 1)

    mu = KMeans(x, n_clusters, kmeans_iters)

    # variances for every class K for every image feature
    vary = vr * np.ones((1, n_clusters)) * 2
    # set all diagonal variances for each cluster
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    logLikelihood = np.zeros((iters, 1))

    # Do iters iterations of EM
    for i in xrange(iters):
        # Do the E step

        # respTot used to calulate p - mixing values
        respTot = np.zeros((n_clusters, 1))
        # used to calculate means of clusters
        respX = np.zeros((N, n_clusters))
        # used to calculate variance of clusters
        respDist = np.zeros((N, n_clusters))
        # inverse variance across class by features
        ivary = 1 / vary

        logNorm = tf.log(p) - 0.5 * N * np.log(2 * np.pi) - \
            0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
        logPcAndx = np.zeros((n_clusters, T))
 
        # for every class
        for k in xrange(n_clusters):
            dis = (x - mu[:, k].reshape(-1, 1))**2
            logPcAndx[k, :] = logNorm[k] - 0.5 * \
                np.sum(ivary[:, k].reshape(-1, 1) * dis, axis=0)

        mx = np.max(logPcAndx, axis=0).reshape(1, -1)
        PcAndx = np.exp(logPcAndx - mx)
        Px = np.sum(PcAndx, axis=0).reshape(1, -1)
        PcGivenx = PcAndx / Px
        logLikelihood[i] = np.sum(np.log(Px) + mx)

        print 'Iter %d logLikelihood %.5f' % (i + 1, logLikelihood[i])

                # Do the M step
        # update mixing coefficients
        respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
        p = respTot

        # update mean
        respX = np.zeros((N, K))
        for k in xrange(K):
            respX[:, k] = np.mean(x * PcGivenx[k, :].reshape(1, -1), axis=1)

        mu = respX / respTot.T

        # update variance
        respDist = np.zeros((N, K))
        for k in xrange(K):
            respDist[:, k] = np.mean(
                (x - mu[:, k].reshape(-1, 1))**2 * PcGivenx[k, :].reshape(1, -1), axis=1)

        vary = respDist / respTot.T
        vary = (vary >= minVary) * vary + (vary < minVary) * minVary

    return p, mu, vary, logLikelihood

def mogLogLikelihood(p, mu, vary, x):
    """ Computes log-likelihood of data under the specified MoG model

    Inputs:
      x: data with one data vector in each column.
      p: probabilities of clusters.
      mu: mean of the clusters, one in each column.
      vary: variances for the cth cluster, one in each column.

    Returns:
      logLikelihood: log-likelihood of data after every iteration.
    """
    K = p.shape[0]
    N, T = x.shape
    ivary = 1 / vary
    logLikelihood = np.zeros(T)
    for t in xrange(T):
        # Compute log P(c)p(x|c) and then log p(x)
        logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
            - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
            - 0.5 * \
            np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)
                   ** 2, axis=0).reshape(-1, 1)

        mx = np.max(logPcAndx, axis=0)
        logLikelihood[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx

    return logLikelihood


def q2():
    # Question 4.2 and 4.3
    K = 7
    iters = 10
    minVary = 0.01
    randConst = 0.1

    # load data
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData(
        '../toronto_face.npz')

    # Train a MoG model with 7 components on all training data, i.e., inputs_train,
    # with both original initialization and kmeans initialization.
    #------------------- Add your code here ---------------------
    p, mu, vary, logLikelihood = mogEM(
        inputs_train,
        K,
        iters,
        randConst=1,
        minVary=minVary
    )

    # show the means per cluster
    ShowMeans(mu, 0)
    # show the variances per cluster
    ShowMeans(vary, 1)

    # print the mixing coefs
    print p

    # print training data
    print logLikelihood


def q4():
    # Question 4.4
    iters = 10
    minVary = 0.01
    randConst = 1.0

    numComponents = np.array([7, 14, 21, 28, 35])
    T = numComponents.shape[0]

    # create arrary to hold number of classification errors for each number of
    # mixture
    errorTrain = np.zeros(T)
    errorTest = np.zeros(T)
    errorValidation = np.zeros(T)

    # convergence values
    convergence_anger = np.zeros(T)
    convergence_happy = np.zeros(T)

    # extract data of class 1-Anger, 4-Happy
    dataQ4 = LoadDataQ4('../toronto_face.npz')
    # images

    x_train = np.concatenate(
        [dataQ4['x_train_anger'], dataQ4['x_train_happy']], axis=1)

    x_valid = np.concatenate(
        [dataQ4['x_valid_anger'], dataQ4['x_valid_happy']], axis=1)

    x_test = np.concatenate(
        [dataQ4['x_test_anger'], dataQ4['x_test_happy']], axis=1)

    # label
    y_train = np.concatenate(
        [dataQ4['y_train_anger'], dataQ4['y_train_happy']])

    y_valid = np.concatenate(
        [dataQ4['y_valid_anger'], dataQ4['y_valid_happy']])

    y_test = np.concatenate([dataQ4['y_test_anger'], dataQ4['y_test_happy']])

    # Hints: this is p(d), use it based on Bayes Theorem
    num_anger_train = dataQ4['x_train_anger'].shape[1]
    num_happy_train = dataQ4['x_train_happy'].shape[1]

    # vector with log(class num / total)
    log_likelihood_class = np.log(
        [num_anger_train, num_happy_train]) - np.log(num_anger_train + num_happy_train)

    # for each different number of component config
    for t in xrange(T):
        # get the number of components
        K = numComponents[t]
        print 'training model with clusters K = %d' % K

        # Train a MoG model with K components
        # Hints: using (x_train_anger, x_train_happy) train 2 MoGs
        #-------------------- Add your code here ------------------------------

        x_train_anger = dataQ4['x_train_anger']
        x_train_happy = dataQ4['x_train_happy']

        print 'modeling training anger with %d clusters' % K
        print '-----------------------------------------'
        # compute MoG on the anger dataset
        anger_model = {}
        p, mu, vary, logLikelihood = mogEM(
            x_train_anger,
            K,
            iters,
            randConst,
            minVary
        )

        anger_model['p'] = p
        anger_model['mu'] = mu
        anger_model['vary'] = vary
        anger_model['logLikelihood'] = logLikelihood

        convergence_anger[t] = logLikelihood[-1]

        print '-----------------------------------------'
        print 'modeling training happy with %d clusters' % K
        print '-----------------------------------------'
        # compute MoG on the happy dataset
        happy_model = {}
        p, mu, vary, logLikelihood = mogEM(
            x_train_happy,
            K,
            iters,
            randConst,
            minVary
        )

        happy_model['p'] = p
        happy_model['mu'] = mu
        happy_model['vary'] = vary
        happy_model['logLikelihood'] = logLikelihood

        convergence_happy[t] = logLikelihood[-1]

        print '-----------------------------------------'
        print '-----------------------------------------'
        print '-----------------------------------------'
        print '-----------------------------------------'
        print '-----------------------------------------'

        # check for accidental mutations
        assert not (happy_model['mu'] == anger_model['mu']).all()

        #------------------- Answers ---------------------

        # Compute the probability P(d|x), classify examples, and compute error rate
        # Hints: using (x_train, y_train), (x_valid, y_valid), (x_test, y_test)
        # to compute error rates, you may want to use mogLogLikelihood function
        #-------------------- Add your code here ------------------------------

        # P(d | x) proprtional to P(x | d) P(d) for a single dataset point
        # P(x | d) is the value of the mixture of gaussian cluster pdf for x point

        dataset = [
            ('train', x_train, y_train, errorTrain),
            ('validation', x_valid, y_valid, errorValidation),
            ('test', x_test, y_test, errorTest)
        ]

        for name, x, y, store in dataset:
            print 'running model with %d clusters on %s dataset' % (K, name)

            # get posteriors of datapoints
            posterior_anger = mogLogLikelihood(
                anger_model['p'], anger_model['mu'], anger_model['vary'], x
            ) + log_likelihood_class[0]

            posterior_happy = mogLogLikelihood(
                happy_model['p'], happy_model['mu'], happy_model['vary'], x
            ) + log_likelihood_class[1]

            # now that we have posteriors we can get the class
            # the classifier thinks datapoints are
            # anger == 0 and happy = 1
            classified_points = (posterior_anger < posterior_happy) * 1

            assert ((classified_points == 1) + (classified_points == 0)).all()
            assert ((y == 1) + (y == 0)).all()

            # now compare to the ground truth
            incorrect_num = np.float(np.sum(classified_points != y))
            store[t] = incorrect_num / y.size

            print '-----------------------------------------'

        #------------------- Answers ---------------------

    print 'FINISHED RUNNING ALL MODELS'

    print errorTrain
    print errorValidation
    print errorTest

