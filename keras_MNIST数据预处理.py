import numpy as np 
import pandas as pd 
from keras.utils import np_utils
from keras.datasets import mnist

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

import matplotlib.pyplot as plt 
def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image, cmap='binary')
    plt.show()

def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num > 25:
        num = 25
    for i in range(0, num):
        ax = plt.subplot(5, 5, i + 1)
        ax.imshow(images[idx], cmap = 'binary')
        title = "label = " + str(labels[idx])
        if len(prediction) > 0:
            title += ", prediction = " + str(prediction[idx])

        ax.set_title(title, fontsize = 10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()

print('x_train_image:', x_train_image.shape)
print('y_train_label:', y_train_label.shape)

# 将image以reshape转换
x_Train = x_train_image.reshape(x_train_image.shape[0], 784).astype('float32')
x_Test = x_test_image.reshape(x_test_image.shape[0], 784).astype('float32')

print('x_train: ', x_Train.shape)
print('x_test: ', x_Test.shape)

print(x_train_image[0])

# 将数字图像images的数字标准化
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

print(x_Train_normalize[0])

print(y_train_label[:5])

# label标签字段进行One-Hot Encoding转换
y_TrainOneHot = np_utils.to_categorical(y_train_label)
y_TestOneHot = np_utils.to_categorical(y_test_label)

print(y_TrainOneHot[:5])