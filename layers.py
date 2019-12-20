import numpy as np
import tensorflow as tf

def expander_fc(inpt, units, activation, sparsity, name):
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
        
        # expander linear
        expand_size = int(units * sparsity)

        mask_w = np.zeros((units, in_dim), dtype=np.float32)
        if units < in_dim:
            for i in range(units):
                x = np.random.permutation(in_dim)
                for j in range(expand_size):
                    mask_w[i][x[j]] = 1.0
        else:
            for i in range(in_dim):
                x = np.random.permutation(units)
                for j in range(expand_size):
                    mask_w[x[j]][i] = 1.0
        
        # transpose the mask
        mask_w = np.transpose(mask_w)
        
        mask_w = tf.get_variable(name='expander_mask',
                                 initializer=mask_w,
                                 dtype=tf.float32,
                                 trainable=False)
        
        tf.add_to_collection('masks', mask_w)

        w = tf.multiply(w, mask_w)

        return activation(tf.add(tf.matmul(inpt, w), b))

def single_rss(inpt, units, sparsity, layer_name):
    with tf.variable_scope(layer_name):
        in_dim = inpt.get_shape().as_list()[-1]
        w = tf.get_variable(name='W',
                            shape=[in_dim, units],
                            initializer=tf.initializers.orthogonal(1.0),
                            trainable=True)
        tf.add_to_collection('weights', w)
        b = tf.get_variable(name='b',
                            shape=[units,],
                            initializer=tf.constant_initializer(0.01))
        mask_w = np.random.binomial(n=1, p=sparsity, size=[in_dim, units]).astype(np.float32)
        # mask_w = tf.convert_to_tensor(mask_w, dtype=tf.float32)
        mask_w = tf.get_variable(name='mask',
                                 initializer=mask_w,
                                 dtype=tf.float32,
                                 trainable=False)
        tf.add_to_collection('rss_masks', mask_w)
        w = tf.multiply(w, mask_w)
        return tf.add(tf.matmul(inpt, w), b)


def rss(inpt, units, activations, sparsity, name):
    with tf.variable_scope(name):
        outs = [inpt]

        num_layers = len(units)
        for n in range(num_layers):
            inter_out = 0.0
            for n_sub in range(n+1):
                layer_name = '_{}_{}'.format((n+1), (n_sub+1))
                divisor = (n+1)
                out = single_rss(inpt=outs[n_sub], units=units[n],
                                 sparsity=sparsity / (n + 1),
                                 layer_name=layer_name)
                inter_out = tf.add(inter_out, out)
            inter_out = activations[n](inter_out)
            outs.append(inter_out)

        for o in outs:
            print(o, o.shape)

        return outs[-1]

def single_rss_path(inpt, units, sparsity, layer_name):
    with tf.variable_scope(layer_name):
        in_dim = inpt.get_shape().as_list()[-1]

        keep_dim = int(in_dim * sparsity)
        # really unoptimized, will work on it LATER
        
        mask_w = np.zeros((in_dim, units), dtype=np.float32)
        for i in range(units):
            r_id = np.random.choice(in_dim, keep_dim, replace=False)
            for r in r_id:
                mask_w[r][i] = 1.0

        w = tf.get_variable(name='W',
                            shape=[in_dim, units],
                            initializer=tf.initializers.orthogonal(1.0),
                            trainable=True)
        tf.add_to_collection('weights', w)

        b = tf.get_variable(name='b',
                            shape=[units,],
                            initializer=tf.constant_initializer(0.01))
        
        mask_w = tf.get_variable(name='mask',
                                 initializer=mask_w,
                                 dtype=tf.float32,
                                 trainable=False)
        tf.add_to_collection('rss_path_masks', mask_w)
        w = tf.multiply(w, mask_w)
        
        out = tf.add(tf.matmul(inpt, w), b)
        return out

def rss_path(inpt, units, activations, sparsity, name):
    with tf.variable_scope(name):
        outs = [inpt]

        num_layers = len(units)
        for n in range(num_layers):
            inter_out = 0.0
            divisor = (n + 1)
            for n_sub in range(n+1):
                layer_name = '_{}_{}'.format((n+1), (n_sub+1))

                out = single_rss_path(inpt=outs[n_sub], units=units[n], sparsity=sparsity / divisor,
                                      layer_name=layer_name)

                inter_out = tf.add(inter_out, out)
                
            inter_out = activations[n](inter_out)
            outs.append(inter_out)

        for o in outs:
            print(o, o.shape)

        return outs[-1]


def fc(inpt, units, activation, prune=None, sparsity=None,
       name='fc'):
    # expander fc has a different structure
    if prune == 'fixed_expander':
        print('''

            expander

            ''')
        return expander_fc(inpt, units, activation, sparsity, name)

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
            print('''

                fixed_random

                ''')
            # mask = np.random.randint(2, size=[in_dim, units])
            mask_w = np.random.binomial(n=1, p=sparsity, size=[in_dim, units]).astype(np.float32)
            # mask_w = tf.convert_to_tensor(mask_w, dtype=tf.float32)
            mask_w = tf.get_variable(name='mask',
                                     initializer=mask_w,
                                     dtype=tf.float32,
                                     trainable=False)
            tf.add_to_collection('masks', mask_w)
            w = tf.multiply(w, mask_w)
        
        elif prune == 'lws':
            mask_w = tf.get_variable(name='mask',
                                     initializer=tf.ones([in_dim, units]),
                                     dtype=tf.float32,
                                     trainable=False)
            tf.add_to_collection('masks', mask_w)
            w = tf.multiply(w, mask_w)

        elif prune == 'fixed_random_wrap':
            # first apply the random mask
            # mask = np.random.randint(2, size=[in_dim, units])
            mask_w = np.random.binomial(n=1, p=sparsity, size=[in_dim, units]).astype(np.float32)
            # mask_w = tf.convert_to_tensor(mask_w, dtype=tf.float32)
            mask_w = tf.get_variable(name='mask',
                                     initializer=mask_w,
                                     dtype=tf.float32,
                                     trainable=False)
            tf.add_to_collection('masks', mask_w)
            w = tf.multiply(w, mask_w)

            # then apply wrap
            out_dim = 9
            kernel_size, stride_size = 3, 1
            wrap_W = tf.get_variable(name='wrap_W',
                                     shape=[kernel_size, kernel_size, 1, out_dim],
                                     dtype=tf.float32,
                                     initializer=tf.initializers.orthogonal(1.0))
            wrap_b = tf.get_variable(name='wrap_b',
                                     shape=[out_dim],
                                     dtype=tf.float32,
                                     initializer=tf.constant_initializer(0.01))
            expanded_W = tf.expand_dims(tf.expand_dims(w, 0), -1)
            conv_W = tf.nn.conv2d(input=expanded_W,
                                  filter=wrap_W,
                                  strides=[1, stride_size, stride_size, 1],
                                  padding='SAME',
                                  name='wrapped_conv2d_')
            conv_W = tf.nn.bias_add(value=conv_W,
                                    bias=wrap_b,
                                    name='wrapped_bias_')
            conv_W = activation(conv_W)
            
            w = tf.squeeze(tf.reduce_max(expanded_W, -1))
            print('shape of wrapped w: {}'.format(w.shape))

        return activation(tf.add(tf.matmul(inpt, w), b))

def expander_conv2d(inputs, 
                    filters, 
                    kernel_size, 
                    stride_size, 
                    padding,
                    kernel_initializer,
                    bias_initializer,
                    normalization,
                    name,
                    sparsity):
    print('''

        expander conv2d

        ''')
    with tf.variable_scope(name):
        in_channels = inputs.get_shape()[-1]
        out_channels = filters
        W = tf.get_variable(name='W', 
                            shape=[kernel_size, kernel_size, in_channels, out_channels],
                            dtype=tf.float32,
                            initializer=kernel_initializer)
        tf.add_to_collection('weights', W)

        n = kernel_size * kernel_size * out_channels
        expand_size = int(filters * sparsity)
        mask_w = np.zeros((1, 1, (in_channels), out_channels), dtype=np.float32)
        if in_channels > out_channels:
            for i in range(out_channels):
                x = np.random.permutation(in_channels)
                for j in range(expand_size):
                    mask_w[0][0][x[j]][i] = 1.0
        else:
            for i in range(in_channels):
                x = np.random.permutation(out_channels)
                for j in range(expand_size):
                    mask_w[0][0][i][x[j]] = 1.0

        mask_w = np.tile(mask_w, (kernel_size, kernel_size, 1, 1))
        mask_w = tf.convert_to_tensor(mask_w)
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
        return conv

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
    
    if prune == 'fixed_skip' or prune == 'fixed_skip_path':
        return expander_conv2d(inputs=inputs, 
                               filters=filters, 
                               kernel_size=kernel_size, 
                               stride_size=stride_size, 
                               padding=padding,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               normalization=normalization,
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

        elif prune == 'lws':
            mask_w = tf.get_variable(name='mask',
                                     initializer=tf.ones([kernel_size, kernel_size, in_channels, out_channels]), 
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