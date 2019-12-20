import os

import numpy as np
import tensorflow as tf

sess = tf.get_default_session

class Trainer(object):
    def __init__(self,
                 model,
                 data,
                 nsamples_mb,
                 prune,
                 sparsity,
                 optimizer,
                 learning_rate,
                 learning_rate_scheduler,
                 nepochs,
                 save_dir,
                 save_freq,
                 log_dir,
                 exp_name):

        ###
        ## trainig related
        ###
        self.nsamples_mb = nsamples_mb
        self.nepochs = nepochs

        # get loggers and exp_name
        self.exp_name = exp_name

        # we will not save any model, not really necessary
        self.save_dir = save_dir
        self.save_freq = save_freq

        # we will use log_dir to collect our results
        self.log_dir = log_dir

        # get data
        train_images, test_images = data.train_images, data.test_images

        if model.architecture == 'fc':
            # flatten inputs if model.architecture if fully connected
            train_images = np.reshape(train_images, [train_images.shape[0], np.prod(train_images.shape[1:])])
            test_images = np.reshape(test_images, [test_images.shape[0], np.prod(test_images.shape[1:])])

        self.train_images = train_images
        self.test_images = test_images
        self.train_labels, self.test_labels = data.train_labels, data.test_labels

        ###
        ## losses
        ###
        self.model = model
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=self.model.y_ph, 
                                                               y_pred=self.model.y_pred)
        self.mean_loss = tf.reduce_mean(loss)

        self.acc, self.acc_op = tf.metrics.accuracy(labels=self.model.y_ph,
                                                    predictions=tf.argmax(self.model.y_pred, 1))
        
        if optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            # we only use adam for now
            raise NotImplementedError()

        # for now learning_rate_scheduler == lambda x: x
        self.learning_rate_scheduler = learning_rate_scheduler

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        print('''

            params

            ''')
        for p in params:
            print(p)


        print('''

            masks

            ''')
        masks = tf.get_collection('masks')
        for m in masks:
            print(m)

        print('''

            weights

            ''')
        weights = tf.get_collection('weights')
        for w in weights:
            print(w)

        gradsandvar = optimizer.compute_gradients(self.mean_loss, params)

        # add zeroed gradients if prune is not None
        self.prune = prune
        self.sparsity = sparsity

        # update batch normalization if used
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.apply_gradients(gradsandvar)

        self.saver = tf.train.Saver()
        sess().run(tf.local_variables_initializer())
        sess().run(tf.global_variables_initializer())

    def train(self):

        train_idx = np.arange(self.train_images.shape[0])
        test_idx = np.arange(self.test_images.shape[0])

        curr_iter = 0
        train_losses = []
        train_accs, test_accs = [], []

        # nbatches_epoch = train_idx.shape[0] // self.nsamples_mb        
        for e in range(self.nepochs):
            print('we are at epoch: {}'.format(e))
            np.random.shuffle(train_idx)

            # one epoch
            for start in range(0, train_idx.shape[0], self.nsamples_mb):
                end = start + self.nsamples_mb
                mbidx = train_idx[start:end]
                l, _ = sess().run([self.mean_loss, self.train_op],
                                   feed_dict={self.model.x_ph: self.train_images[mbidx],
                                              self.model.y_ph: self.train_labels[mbidx]})

                # added it to losses
                train_losses.append(l)

            # more accurate accuracy calculations
            if e % (self.nepochs // 5) == 0:
                train_acc = 0.0
                for start in range(0, train_idx.shape[0], self.nsamples_mb):
                    end = start + self.nsamples_mb
                    mbidx = train_idx[start:end]
                    acc, _ = sess().run([self.acc, self.acc_op],
                                         feed_dict={self.model.x_ph: self.train_images[mbidx],
                                                    self.model.y_ph: self.train_labels[mbidx]})
                    train_acc += acc * mbidx.shape[0]
                train_accs.append(train_acc / train_idx.shape[0])
                
                test_acc = 0.0
                for start in range(0, test_idx.shape[0], self.nsamples_mb):
                    end = start + self.nsamples_mb
                    mbidx = test_idx[start:end]
                    acc, _ = sess().run([self.acc, self.acc_op],
                                         feed_dict={self.model.x_ph: self.test_images[mbidx],
                                                    self.model.y_ph: self.test_labels[mbidx]})
                    test_acc += acc * mbidx.shape[0]
                test_accs.append(test_acc / test_idx.shape[0])

        log_file = os.path.join(self.log_dir, self.exp_name)
        np.savez_compressed(log_file, train_losses=np.asarray(train_losses),
                                      train_accs=np.asarray(train_accs),
                                      test_accs=np.asarray(test_accs))

        # self.saver.save(sess(), log_file)