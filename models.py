import numpy as np
import tensorflow as tf

from layers import fc, conv2d, rss, rss_path

class LeNetFC(object):
    def __init__(self,
                 prune,
                 sparsity,
                 input_shape):

        # type of architecture
        self.architecture = 'fc'

        # this is dummy model to test running process
        self.x_ph = tf.placeholder(tf.float32, shape=[None, np.prod(input_shape)], name='x_ph')
        self.y_ph = tf.placeholder(tf.uint8, shape=[None,], name='y_ph')
        
        # dummy mode
        self.mode = tf.placeholder(tf.bool, name='mode')

        with tf.variable_scope('net'):
            if prune == 'fixed_skip':
                out = self.x_ph
                out = rss(out,
                          units=[300, 100, 10],
                          activations=[tf.nn.relu, tf.nn.relu, tf.nn.softmax],
                          sparsity=sparsity,
                          name='rss')
            elif prune == 'fixed_skip_path':
                out = self.x_ph
                out = rss_path(out,
                               units=[300, 100, 10],
                               activations=[tf.nn.relu, tf.nn.relu, tf.nn.softmax],
                               sparsity=sparsity,
                               name='rss_path')
            else:
                out = fc(self.x_ph,
                         units=300, 
                         activation=tf.nn.relu,
                         prune=prune,
                         sparsity=sparsity,
                         name='layer_1')
                out = fc(out,
                         units=100, 
                         activation=tf.nn.relu,
                         prune=prune,
                         sparsity=sparsity,
                         name='layer_2')
                out = fc(out,
                         units=10,
                         activation=tf.nn.softmax,
                         prune=prune,
                         sparsity=sparsity,
                         name='layer_3')
            
            self.y_pred = out

class LeNet5(object):
    def __init__(self,
                 prune,
                 sparsity,
                 input_shape):

        # type of architecture
        self.architecture = 'cnn'

        self.x_ph = tf.placeholder(tf.float32, shape=[None, ] + input_shape, name='x_ph')
        self.y_ph = tf.placeholder(tf.uint8, shape=[None, ], name='y_ph')
        self.mode = tf.placeholder(tf.bool, name='mode')

        with tf.variable_scope('net'):
            out = conv2d(inputs=self.x_ph,
                         filters=6,
                         kernel_size=5,
                         stride_size=1,
                         prune=prune,
                         sparsity=sparsity,
                         name='conv2d_1')
            out = tf.nn.tanh(out)
            out = tf.layers.average_pooling2d(inputs=out,
                                              pool_size=2,
                                              strides=2,
                                              padding='SAME',
                                              name='pool_1')
            out = conv2d(inputs=out,
                         filters=16,
                         kernel_size=5,
                         stride_size=1,
                         prune=prune,
                         sparsity=sparsity,
                         name='conv2d_2')
            out = tf.nn.tanh(out)
            out = tf.layers.average_pooling2d(inputs=out,
                                              pool_size=2,
                                              strides=2,
                                              padding='SAME',
                                              name='pool_2')
            out = tf.layers.flatten(out)
            
            if prune == 'fixed_random' or prune == 'fixed_random_wrap':
                out = fc(out,
                         units=120, 
                         activation=tf.nn.tanh,
                         prune=prune,
                         sparsity=sparsity,
                         name='layer_1')
                
                out = fc(out,
                         units=84, 
                         activation=tf.nn.tanh,
                         prune=prune,
                         sparsity=sparsity,
                         name='layer_2')
                
                out = fc(out,
                         units=10,
                         activation=tf.nn.softmax,
                         prune=prune,
                         sparsity=sparsity,
                         name='layer_3')

            elif prune == 'fixed_skip':
                out = rss(out,
                          units=[120, 84, 10],
                          activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.softmax],
                          sparsity=sparsity,
                          name='rss')
            elif prune == 'fixed_skip_path':
                out = rss_path(out,
                          units=[120, 84, 10],
                          activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.softmax],
                          sparsity=sparsity,
                          name='rss')

            self.y_pred = out


class AlexNetS(object):
    def __init__(self,
                 prune,
                 sparsity,
                 input_shape):

        # type of architecture
        self.architecture = 'cnn'

        self.x_ph = tf.placeholder(tf.float32, shape=[None, ] + input_shape, name='x_ph')
        self.y_ph = tf.placeholder(tf.uint8, shape=[None, ], name='y_ph')
        self.mode = tf.placeholder(tf.bool, name='mode')

        with tf.variable_scope('net'):
            out = conv2d(inputs=self.x_ph,
                         filters=96,
                         kernel_size=11,
                         stride_size=2,
                         prune=prune,
                         sparsity=sparsity,
                         name='conv2d_1')
            out = tf.nn.relu(out)

            out = conv2d(inputs=out,
                         filters=256,
                         kernel_size=5,
                         stride_size=2,
                         prune=prune,
                         sparsity=sparsity,
                         name='conv2d_2')
            out = tf.nn.relu(out)

            out = conv2d(inputs=out,
                         filters=384,
                         kernel_size=3,
                         stride_size=2,
                         prune=prune,
                         sparsity=sparsity,
                         name='conv2d_3')
            out = tf.nn.relu(out)

            out = conv2d(inputs=out,
                         filters=384,
                         kernel_size=3,
                         stride_size=2,
                         prune=prune,
                         sparsity=sparsity,
                         name='conv2d_4')
            out = tf.nn.relu(out)

            out = conv2d(inputs=out,
                         filters=256,
                         kernel_size=3,
                         stride_size=2,
                         prune=prune,
                         sparsity=sparsity,
                         name='conv2d_5')
            out = tf.nn.relu(out)

            out = tf.layers.flatten(out)
            print('''

                out.shape: {}

                '''.format(out.shape))
            if prune == 'fixed_random' or prune == 'fixed_random_wrap':
                out = fc(out,
                         units=512, 
                         activation=tf.nn.relu,
                         prune=prune,
                         sparsity=sparsity,
                         name='layer_1')
                
                out = fc(out,
                         units=512, 
                         activation=tf.nn.relu,
                         prune=prune,
                         sparsity=sparsity,
                         name='layer_2')
                
                out = fc(out,
                         units=10,
                         activation=tf.nn.softmax,
                         prune=prune,
                         sparsity=sparsity,
                         name='layer_3')

            elif prune == 'fixed_skip':
                out = rss(inpt,
                          units=[512, 512, 10],
                          activations=[tf.nn.relu, tf.nn.relu, tf.nn.softmax],
                          sparsity=sparsity,
                          name='rss')
            
            elif prune == 'fixed_skip_path':
                out = rss_path(inpt,
                          units=[512, 512, 10],
                          activations=[tf.nn.relu, tf.nn.relu, tf.nn.softmax],
                          sparsity=sparsity,
                          name='rss')
            
            self.y_pred = out
            
# do NOT forget to use weight_initializer, bias_initializer,
# prune and sparsity
def model_builder(architecture,
                  prune,
                  sparsity,
                  input_shape):
    # prune is only effective at this level iff fixed    
    if architecture == 'lenet-300-100':
        return LeNetFC(prune,
                       sparsity,
                       input_shape)

    elif architecture == 'lenet-5':
        return LeNet5(prune,
                      sparsity,
                      input_shape)

    elif architecture == 'alexnet-s':
        return AlexNetS(prune,
                        sparsity,
                        input_shape)
    else:
        raise NotImplementedError()