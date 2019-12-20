#!/bin/bash

bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-2-wrap --sparsity 0.02 --prune fixed_random_wrap --nepochs 50 --nsamples_mb 128"
bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-10-wrap --sparsity 0.1 --prune fixed_random_wrap --nepochs 50 --nsamples_mb 128"

bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-2-wrap-c --sparsity 0.02 --prune fixed_random_wrap_c --nepochs 50 --nsamples_mb 128"
bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-10-wrap-c --sparsity 0.1 --prune fixed_random_wrap_c --nepochs 50 --nsamples_mb 128"