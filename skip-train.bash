#!/bin/bash
bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rskip-2 --sparsity 0.02 --prune fixed_skip --nepochs 50 --nsamples_mb 128"
bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rskip-10 --sparsity 0.1 --prune fixed_skip --nepochs 50 --nsamples_mb 128"

bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rskippath-2 --sparsity 0.02 --prune fixed_skip_path --nepochs 50 --nsamples_mb 128"
bsub -n 8 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-300-100 --exp_name lenet-CIFAR10-rskippath-10 --sparsity 0.1 --prune fixed_skip_path --nepochs 50 --nsamples_mb 128"

bsub -n 16 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-5 --exp_name lenet-5-CIFAR10-rskippath-2 --sparsity 0.02 --prune fixed_skip_path --nepochs 50 --nsamples_mb 128"
bsub -n 16 -W 8:00 "python3 main.py --dataset cifar10 --architecture lenet-5 --exp_name lenet-5-CIFAR10-rskippath-10 --sparsity 0.1 --prune fixed_skip_path --nepochs 50 --nsamples_mb 128"

bsub -n 16 -R "rusage[ngpus_excl_p=1]" "python3 main.py --dataset cifar10 --architecture alexnet-s --exp_name alexnet-CIFAR10-rskippath-2 --sparsity 0.02 --prune fixed_skip_path --nepochs 100"
bsub -n 16 -R "rusage[ngpus_excl_p=1]" "python3 main.py --dataset cifar10 --architecture alexnet-s --exp_name alexnet-CIFAR10-rskippath-10 --sparsity 0.1 --prune fixed_skip_path --nepochs 100"