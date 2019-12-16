import numpy as np
import tensorflow as tf

def fc(inpt, units, activation, prune=None, sparsity=None,
       name='fc'):
    # expander fc has a different structure
    if prune == 'fixed_expander':
        return expander_fc(out, units, sparsity, name)

    with tf.variable_scope(name):
        in_dim = inpt.get_shape().as_list()[-1]
        w = tf.get_variable(name='W',
                            shape=[in_dim, units],
                            initializer=tf.initializers.orthogonal(1.0),
                            trainable=True)
        tf.add_to_collection('weights', w)
        b = tf.get_variable(name='b',
                            shape=[units,],
                            initializer=tf.constant_initializer(0.01))
        
        if prune == 'fixed_random':
            # mask = np.random.randint(2, size=[in_dim, units])
            mask_w = np.random.binomial(n=1, p=sparsity, size=[in_dim, units]).astype(np.float32)
            # mask_w = tf.convert_to_tensor(mask_w, dtype=tf.float32)
            mask_w = tf.get_variable(name='mask',
                                     initializer=mask_w,
                                     dtype=tf.float32,
                                     trainable=False)
            tf.add_to_collection('masks', mask_w)
            w = tf.multiply(w, mask_w)

        return activation(tf.add(tf.matmul(inpt, w), b))

def conv2d(inputs, 
           filters, 
           kernel_size, 
           stride_size, 
           padding='SAME',
           kernel_initializer=tf.initializers.orthogonal(1.0),
           bias_initializer=tf.constant_initializer(value=0.01),
           normalization=None,
           name='conv2d',
           prune=None,
           sparsity=None):
    
    if prune == 'fixed_expander':
        return expanded_conv2d(inputs=inputs, 
                               filters=filters, 
                               kernel_size=kernel_size, 
                               stride_size=stride_size, 
                               padding=padding,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               normalization=normalization,
                               activation=activation,
                               name=name,
                               sparsity=sparsity)

    with tf.variable_scope(name):
        in_channels = inputs.get_shape()[-1]
        out_channels = filters
        W = tf.get_variable(name='W', 
                            shape=[kernel_size, kernel_size, in_channels, out_channels],
                            dtype=tf.float32,
                            initializer=kernel_initializer)
        tf.add_to_collection('weights', W)

        if prune == 'fixed_random':
            mask_w = np.random.binomial(n=1, p=sparsity,
                                      size=[kernel_size, kernel_size, in_channels, out_channels]).astype(np.float32)
            mask_w = tf.get_variable(name='mask',
                                     initializer=mask_w, 
                                     dtype=tf.float32,
                                     trainable=False)
            tf.add_to_collection('masks', mask_w)
            W = tf.multiply(W, mask_w)

        b = tf.get_variable(name='b', 
                            shape=[out_channels], 
                            dtype=tf.float32,
                            initializer=bias_initializer)
        
        conv = tf.nn.conv2d(input=inputs, 
                            filter=W, 
                            strides=[1, stride_size, stride_size, 1], 
                            padding=padding,
                            name='conv2d_')
        conv = tf.nn.bias_add(value=conv, 
                              bias=b,
                              name='bias_')

        if normalization:
            raise NotImplementedError()
            '''
            if normalization == 'instance_normalization':
                conv = instance_normalization(inputs=conv,
                                              mode=mode,
                                              name='instance_normalization')
            elif normalization == 'batch_normalization':
                conv = tf.layers.batch_normalization(conv, training=True)
            '''

        '''
        # we might have pooling layers
        # apply activation while building the model
        # only works for some activation functions
        # need to change this for general case
        if activation:
            return activation(features=conv, 
                              name='activation')
        '''

        return conv