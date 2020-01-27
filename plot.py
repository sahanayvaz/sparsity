import os
import numpy as np
import matplotlib.pyplot as plt


def sliding_mean(data_array, window=10):  
    # data_array = np.array(data_array)  
    new_list = []
    for i in range(data_array.shape[0]):  
        indices = range(max(i - window + 1, 0),  
                        min(i + window + 1, data_array.shape[0]))  
        avg = 0
        for j in indices:
            avg += data_array[j]  
        avg /= float(len(indices))  
        new_list.append(avg)
    return np.asarray(new_list)

m = 'lenet'
d = 'CIFAR10'

'''
check_results = ['{}-{}-dense'.format(m, d),
                 '{}-{}-rs-10'.format(m, d),
                 '{}-{}-rs-2'.format(m, d)]
'''

check_results = ['{}-{}-dense'.format(m, d),
                 '{}-{}-rs-10'.format(m, d),
                 '{}-{}-rs-2'.format(m, d),
                 '{}-{}-rs-10-wrap'.format(m, d),
                 '{}-{}-rs-2-wrap'.format(m, d),
                 '{}-{}-rs-10-wrap-c'.format(m, d),
                 '{}-{}-rs-2-wrap-c'.format(m, d)]

'''
'{}-{}-rskip-10'.format(m, d),
                 '{}-{}-rskip-2'.format(m, d),
'{}-{}-rskippath-10'.format(m, d),
                 '{}-{}-rskippath-2'.format(m, d)
'''

plt.figure(figsize=(12, 9))
plotHandles = []
labels = []
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

for c in check_results:
    results_dir = os.path.join('./log_dir', c)
    results_npz = os.listdir(results_dir)
    test_acc = []
    train_acc = []
    train_losses = []
    for r_npz in results_npz:
        if 'npz' in r_npz:
            full_r_npz = os.path.join(results_dir, r_npz)
            r = np.load(full_r_npz)
            # print(r['train_losses'].shape, r['train_accs'].shape, r['test_accs'].shape)
            test_acc.append(r['test_accs'])
            train_acc.append(r['train_accs'])
            train_losses.append(r['train_losses'])

    test_acc = np.asarray(test_acc)
    train_acc = np.asarray(train_acc)
    train_losses = np.asarray(train_losses)
    mean = sliding_mean(np.mean(train_losses, 0))
    std = sliding_mean(np.std(train_losses, 0))
    iterations = np.arange(mean.shape[0])
    
    x, = plt.plot(iterations, mean)
    plt.fill_between(iterations, mean - std, mean + std, alpha=0.1)
    plotHandles.append(x)
    labels.append(c)

    print(c, 100 * (1.0 - np.mean(test_acc, 0))[-1], np.std(test_acc, 0)[-1], 100 * (1.0 - np.mean(train_acc, 0))[-1], np.std(train_acc, 0)[-1])

plt.legend(plotHandles, labels, loc='upper right', ncol=1)
plt.savefig('./visuals/{}-{}-wrap-more.png'.format(m, d))
plt.show()