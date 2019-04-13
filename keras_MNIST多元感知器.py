# 导入所需模块
from keras.utils import np_utils
import numpy as np 
np.random.seed(10)

# 读取MNIST数据
from keras.datasets import mnist
(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

# 预处理
x_Train = x_train_image.reshape(x_train_image.shape[0], 784).astype('float32')
x_Test = x_test_image.reshape(x_test_image.shape[0], 784).astype('float32')

x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_One_Hot = np_utils.to_categorical(y_test_label)

# 建立模型
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# 建立Sequential模型
model = Sequential()

# 建立输入层与隐藏层
'''
units = 256: 定义隐藏层神经元个数为256个
input_dim = 784: 设置输入层神经元个数为784
kernel_initializer = 'normal': 使用normal distribution正态分布的随机数来初始化weight（权重）和bias（偏置）
activation: 定义激活函数为relu
'''
model.add(Dense(units=1000, input_dim = 784, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1000, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))

# 建立输出层
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))

# 查看模型的摘要
print(model.summary())

# 进行训练
'''
loss: 设置损失函数
optimizer: 设置优化器
metrics: 设置评估模型的方式
'''
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 开始训练
'''
validation_split = 0.2: 训练之前Keras会自动将数据分成80%训练数据和20%验证数据。
verbose=2: 显示训练过程
'''
train_history = model.fit(x=x_Train_normalize, y=y_Train_OneHot, validation_split=0.2, epochs=10, batch_size=200, verbose=2)

# 建立show_train_history显示训练过程
import matplotlib.pyplot as plt 
def show_train_history(train_history, train, validation):
    '''
    输入参数：之前训练过程所产生的train_history；训练数据的执行结果；验证数据的执行结果
    '''
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history, 'acc', 'val_acc')
show_train_history(train_history, 'loss', 'val_loss')

# 以测试数据评估模型准确率
scores = model.evaluate(x_Test_normalize, y_Test_One_Hot)
print()
print('accuracy = ', scores[1])

# # 进行预测
# prediction = model.predict_classes(x_Test)
# print(prediction)

# def plot_images_labels_prediction(images, labels, prediction, idx, num = 10):
#     fig = plt.gcf()
#     fig.set_size_inches(12, 14)
#     if num > 25:
#         num = 25
#     for i in range(0, num):
#         ax = plt.subplot(5, 5, i + 1)
#         ax.imshow(images[idx], cmap = 'binary')
#         title = "label = " + str(labels[idx])
#         if len(prediction) > 0:
#             title += ", prediction = " + str(prediction[idx])

#         ax.set_title(title, fontsize = 10)
#         ax.set_xticks([])
#         ax.set_yticks([])
#         idx += 1
#     plt.show()

# # plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx=340)

# # 显示混淆矩阵
# import pandas as pd 
# print(pd.crosstab(y_test_label, prediction, rownames=['label'], colnames=['predict']))

# # 建立真实值与预测DataFrame
# df = pd.DataFrame({'label': y_test_label, 'predict': prediction})
# print(df)
# # 查询真实值是“5”但预测值是“3”的数据
# print(df[(df.label==5)&(df.predict==3)])