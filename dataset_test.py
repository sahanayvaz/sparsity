import matplotlib.pyplot as plt
from tensorflow.keras import datasets
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
print(np.max(train_images), np.min(train_images))
print(train_labels.shape, train_labels[0])
train_images, test_images = train_images / 255.0, test_images / 255.0

print(train_images.shape, train_labels.shape)

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    # plt.xlabel(class_names[train_labels[i][0]])
plt.show()
