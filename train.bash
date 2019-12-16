#!/bin/bash

bsub -n 8 "python3 main.py --dataset mnist --architecture lenet-300-100 --exp_name lenet-MNIST-dense"
bsub -n 8 "python3 main.py --dataset mnist --architecture lenet-300-100 --exp_name lenet-MNIST-rs-2 --sparsity 0.02 --prune fixed_random"
bsub -n 8 "python3 main.py --dataset mnist --architecture lenet-300-100 --exp_name lenet-MNIST-rs-10 --sparsity 0.1 --prune fixed_random"

bsub -n 8 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-dense --nepochs 50 --nsamples_mb 128"
bsub -n 8 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-2 --sparsity 0.02 --prune fixed_random --nepochs 50 --nsamples_mb 128"
bsub -n 8 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-10 --sparsity 0.1 --prune fixed_random --nepochs 50 --nsamples_mb 128"

bsub -n 16 "python3 main.py --dataset mnist --architecture lenet-5 --exp_name lenet-5-MNIST-dense"
bsub -n 16 "python3 main.py --dataset mnist --architecture lenet-5 --exp_name lenet-5-MNIST-rs-2 --sparsity 0.02 --prune fixed_random"
bsub -n 16 "python3 main.py --dataset mnist --architecture lenet-5 --exp_name lenet-5-MNIST-rs-10 --sparsity 0.1 --prune fixed_random"

bsub -n 16 "python3 main.py --dataset cifar10 --architecture lenet-5 --exp_name lenet-5-CIFAR10-dense --nepochs 50 --nsamples_mb 128"
bsub -n 16 "python3 main.py --dataset cifar10 --architecture lenet-5 --exp_name lenet-5-CIFAR10-rs-2 --sparsity 0.02 --prune fixed_random --nepochs 50 --nsamples_mb 128"
bsub -n 16 "python3 main.py --dataset cifar10 --architecture lenet-5 --exp_name lenet-5-CIFAR10-rs-10 --sparsity 0.1 --prune fixed_random --nepochs 50 --nsamples_mb 128"

bsub -n 16 -R "rusage[ngpus_excl_p=1]" "python3 main.py --dataset mnist --architecture alexnet-s --exp_name alexnet-MNIST-dense --nepochs 20"
bsub -n 16 -R "rusage[ngpus_excl_p=1]" "python3 main.py --dataset mnist --architecture alexnet-s --exp_name alexnet-MNIST-rs-2 --sparsity 0.02 --prune fixed_random --nepochs 20"
bsub -n 16 -R "rusage[ngpus_excl_p=1]" "python3 main.py --dataset mnist --architecture alexnet-s --exp_name alexnet-MNIST-rs-10 --sparsity 0.1 --prune fixed_random --nepochs 20"

bsub -n 16 -R "rusage[ngpus_excl_p=1]" "python3 main.py --dataset cifar10 --architecture alexnet-s --exp_name alexnet-CIFAR10-dense --nepochs 100 --nsamples_mb 128"
bsub -n 16 -R "rusage[ngpus_excl_p=1]" "python3 main.py --dataset cifar10 --architecture alexnet-s --exp_name alexnet-CIFAR10-rs-2 --sparsity 0.02 --prune fixed_random --nepochs 100 --nsamples_mb 128"
bsub -n 16 -R "rusage[ngpus_excl_p=1]" "python3 main.py --dataset cifar10 --architecture alexnet-s --exp_name alexnet-CIFAR10-rs-10 --sparsity 0.1 --prune fixed_random --nepochs 100 --nsamples_mb 128"
