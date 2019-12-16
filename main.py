'''
main.py to specify configs for train.py
'''

import os
import argparse

import tensorflow as tf
import csv
import deepdish

import utils
from loader import Loader
from models import model_builder
from train import Trainer

class Logger(object):
    def __init__(self, log_dir, exp_name, train_test):
        csv_path = os.path.join(log_dir, 
                                '{}-results-{}.csv'.format(exp_name, train_test))
        self._file = open(csv_path, 'w', newline='')
        self.writer = csv.writer(self._file)
    def write_row(self, row):
        self.writer.writerow(row)

    def close_file(self):
        self._file.close()

def set_experiment_vars(seed):
    # set global seeds
    utils.set_global_seeds(seed) 
    return utils.setup_tensorflow_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    '''
    # training related
    dataset: dataset to train (mnist, cifar10 or tinyimagenet)
    architecture: model architecture (lenet-300-100, lenet-5, alexnet-b1)
    sparsity: level of sparsity, 1.0 == no sparsity (default), 
                                 0.10 == prune %90 of weights
    prune: how to prune (none, 
                         fixed_random, fixed_skipped, fixed_expander, 
                         lws, snip,
                         weighted_lws, weighted_snip)

    # optimization related
    weight_initializer: type of weight initializer
    bias_initializer: type of bias initializer
    optimizer: type of optimizer, default: adam
    learning_rate: learning rate
    learning_rate_scheduler: learning rate schedule (none or linear),
                             default: none for adam
    nsamples_mb: number of samples for a minibatch
    niters: number of iterations
    nexps: number of experiments

    # dataset related
    augment: (bool) whether to augment the dataset or not, default: 0

    # save/load and log related
    save_dir: directory to save model (should come from exp_name)
    save_freq: niters // save_freq gives the save iteration
    log_dir: directory to log results (should come from exp_name)

            --> we will be reporting train/test dataset accuracies
                with mean and std of all runs for an experiment
    exp_name: experiment name
    '''

    # training related
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--architecture', type=str, default='lenet-300-100')
    parser.add_argument('--sparsity', type=float, default=1.0)
    parser.add_argument('--prune', type=str, default=None)

    # optimization related
    # we are currently NOT using weight_initialization, it's always orthogonal
    # parser.add_argument('--weight_initializer', type=str, default='tf.initializers.orthogonal')
    # parser.add_argument('--bias_initializer', type=str, default='tf.initializers.zeros')


    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--learning_rate_scheduler', type=str, default=None)
    parser.add_argument('--nsamples_mb', type=int, default=100)
    parser.add_argument('--nepochs', type=int, default=10)
    parser.add_argument('--nexps', type=int, default=5)

    # dataset related
    parser.add_argument('--augment', type=str, default=None)

    # save/load and log
    # parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_freq', type=int, default=2)
    # parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)

    args = parser.parse_args()

    # dummy check for NOT forgetting
    # i make those mistakes a lot
    if args.sparsity == 1.0:
        assert args.prune is None

    # i made this mistake
    if args.sparsity != 1.0:
        assert args.prune is not None

    # another assertion to define exp_name
    assert args.exp_name is not None

    save_dir = os.path.join('./save_dir', args.exp_name)
    os.makedirs(save_dir, exist_ok=True)

    log_dir = os.path.join('./log_dir', args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    '''
    train_logger = Logger(log_dir=log_dir,
                          exp_name=args.exp_name,
                          train_test='train')

    test_logger = Logger(log_dir=log_dir,
                         exp_name=args.exp_name,
                         train_test='test')
    '''

    # init operation (fix seeds, start session, get logging ready)
    for i in range(args.nexps):
        print('''

            runnign exp: {}

            '''.format(i))

        # do not forget to reset the session
        sess = set_experiment_vars(seed=i)

        # with automatically closes the session
        with sess:

            # handle loading dataset efficiently
            data = Loader(dataset=args.dataset,
                          nsamples_mb=args.nsamples_mb,
                          augment=args.augment)

            # get the model
            # we have two types of sparsity: (fixed and not)
            # for fixed, we build it in the model level
            # for not, we build it iteratively while training
            model = model_builder(architecture=args.architecture,
                                  prune=args.prune,
                                  sparsity=args.sparsity,
                                  input_shape=data.input_shape)

            if args.learning_rate_scheduler is None:
                learning_rate_scheduler = lambda x: x
            else:
                raise NotImplementedError()

            # get the trainer, build losses, sessions, etc
            trainer = Trainer(model=model,
                              data=data,
                              nsamples_mb=args.nsamples_mb,
                              prune=args.prune,
                              sparsity=args.sparsity,
                              optimizer=args.optimizer,
                              learning_rate=args.learning_rate,
                              learning_rate_scheduler=learning_rate_scheduler,
                              nepochs=args.nepochs,
                              save_dir=save_dir,
                              save_freq=args.save_freq,
                              log_dir=log_dir,
                              exp_name='{}-{}'.format(args.exp_name,
                                                      i))

            # trains the model and writes results to loggers
            # save the model to save_dir for iterations defined by
            # (niters // save_freq)
            trainer.train()
        sess.close()
        tf.reset_default_graph()