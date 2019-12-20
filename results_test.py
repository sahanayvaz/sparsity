import os
import numpy as np
import tensorflow as tf

'''
m = 'lenet'
d = 'MNIST'
check_results = ['{}-{}-dense'.format(m, d),
                 '{}-{}-rs-10'.format(m, d),
                 '{}-{}-rs-2'.format(m, d)]
for c in check_results:
'''

c = 'lenet-MNIST-rs-10-wrap'
results_dir = os.path.join('./log_dir', c)

results_npz = os.listdir(results_dir)

test_acc = []
for r_npz in results_npz:
    if 'npz' in r_npz:
        full_r_npz = os.path.join(results_dir, r_npz)
        r = np.load(full_r_npz)
        print(r['train_losses'].shape, r['train_accs'].shape, r['test_accs'].shape)
        test_acc.append(r['test_accs'])

test_acc = np.asarray(test_acc)

print(c)
print(100 * (1.0 - np.mean(test_acc, 0)))
print(np.std(test_acc, 0))