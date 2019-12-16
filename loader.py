'''
data loading pipeline for efficient input/output
'''

import os
import numpy as np
from tensorflow.python.keras.datasets.cifar import load_batch

def load_data():
    dirname = 'cifar-10-batches-py'
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    # path = os.path.join('/cluster/home/sayvaz', 'sparsity', dirname)
    path = dirname
    
    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
        y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)


class Loader(object):
    def __init__(self,
                 dataset,
                 nsamples_mb,
                 augment):
        ###
        ## prepare data
        ###
        if dataset == 'mnist':
            # mnist_path = os.path.join('/cluster/home/sayvaz', 'sparsity', 'mnist.npz')
            mnist_path = 'mnist.npz'

            data = np.load(mnist_path)
            (train_images, train_labels) = data['x_train'], data['y_train']
            (test_images, test_labels) = data['x_test'], data['y_test']
            train_images, test_images = train_images / 255.0, test_images / 255.0
            
            # this is how SNIP does it
            mean, std = np.mean(train_images), np.std(train_images)
            train_images = (train_images - mean) / std
            test_images = (test_images - mean) / std
            train_images = np.expand_dims(train_images, -1)
            test_images = np.expand_dims(test_images, -1)

        elif dataset == 'cifar10':
            (train_images, train_labels), (test_images, test_labels) = load_data()
            train_images, test_images = train_images / 255.0, test_images / 255.0
            
            # this is how SNIP does it    
            channel_mean = np.mean(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
            channel_std = np.std(train_images, axis=(0,1,2), dtype=np.float32, keepdims=True)
            train_images = (train_images - channel_mean) / channel_std
            test_images = (test_images - channel_mean) / channel_std
            train_labels, test_labels = np.squeeze(train_labels), np.squeeze(test_labels)

        if augment:
            raise NotImplementedError()

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

        self.input_shape = list(self.train_images.shape[1:])
