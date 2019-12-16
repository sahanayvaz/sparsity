'''
data loading pipeline for efficient input/output
'''

import numpy as np
from tensorflow.keras import datasets

class Loader(object):
    def __init__(self,
                 dataset,
                 nsamples_mb,
                 augment):
        ###
        ## prepare data
        ###
        if dataset == 'mnist':
            (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
            train_images, test_images = train_images / 255.0, test_images / 255.0
            
            # this is how SNIP does it
            mean, std = np.mean(train_images), np.std(train_images)
            train_images = (train_images - mean) / std
            test_images = (test_images - mean) / std
            train_images = np.expand_dims(train_images, -1)
            test_images = np.expand_dims(test_images, -1)

        elif dataset == 'cifar10':
            (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
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
