#!/bin/bash


bsub -n 8 "python3 main.py --dataset mnist --architecture lenet-300-100 --exp_name lenet-MNIST-rs-2-wrap --sparsity 0.02 --prune fixed_random_wrap --out_dim 1 --kernel_size 9"
bsub -n 8 "python3 main.py --dataset mnist --architecture lenet-300-100 --exp_name lenet-MNIST-rs-10-wrap --sparsity 0.1 --prune fixed_random_wrap --out_dim 1 --kernel_size 9"

bsub -n 8 "python3 main.py --dataset mnist --architecture lenet-300-100 --exp_name lenet-MNIST-rs-2-wrap-18 --sparsity 0.02 --prune fixed_random_wrap --out_dim 1 --kernel_size 18"
bsub -n 8 "python3 main.py --dataset mnist --architecture lenet-300-100 --exp_name lenet-MNIST-rs-10-wrap-18 --sparsity 0.1 --prune fixed_random_wrap --out_dim 1 --kernel_size 18"


bsub -n 8 "python3 main.py --dataset mnist --architecture lenet-300-100 --exp_name lenet-MNIST-rs-2-wrap-c --sparsity 0.02 --prune fixed_random_wrap --out_dim 9 --kernel_size 3"
bsub -n 8 "python3 main.py --dataset mnist --architecture lenet-300-100 --exp_name lenet-MNIST-rs-10-wrap-c --sparsity 0.1 --prune fixed_random_wrap --out_dim 9 --kernel_size 3"


bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-2-wrap --sparsity 0.02 --prune fixed_random_wrap --out_dim 1 --kernel_size 9 --nepochs 50 --nsamples_mb 128"
bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-10-wrap --sparsity 0.1 --prune fixed_random_wrap  --out_dim 1 --kernel_size 9 --nepochs 50 --nsamples_mb 128"


bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-2-wrap-18 --sparsity 0.02 --prune fixed_random_wrap --out_dim 1 --kernel_size 18 --nepochs 50 --nsamples_mb 128"
bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-10-wrap-18 --sparsity 0.1 --prune fixed_random_wrap  --out_dim 1 --kernel_size 18 --nepochs 50 --nsamples_mb 128"


bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-2-wrap-c --sparsity 0.02 --prune fixed_random_wrap --out_dim 9 --kernel_size 3 --nepochs 50 --nsamples_mb 128"
bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rs-10-wrap-c --sparsity 0.1 --prune fixed_random_wrap  --out_dim 9 --kernel_size 3 --nepochs 50 --nsamples_mb 128"