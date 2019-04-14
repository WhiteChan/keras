import numpy as np 
import tensorflow as tf 
'''
X = tf.Variable([[0.4, 0.2, 0.4]])
W = tf.Variable([[-0.5, -0.2], [-0.3, 0.4], [-0.5, 0.2]])
b = tf.Variable([[0.1, 0.2]])

XWb = tf.matmul(X, W) + b
y = tf.nn.sigmoid(tf.matmul(X, W) + b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb: ')
    print(sess.run(XWb))
    print('y: ')
    print(sess.run(y))

# 正态分布
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.Variable([[0.4, 0.2, 0.4]])

y = tf.nn.relu(tf.matmul(X, W) + b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('b: ', sess.run(b))
    print('W: ', sess.run(W))
    print('y: ', sess.run(y))
    (_b, _W, _y) = sess.run((b, W, y))
    print('b: ', _b)
    print('W: ', _W)
    print('y: ', _y)

# 以placeholder传入X值
W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([1, 2]))
X = tf.placeholder("float", [None, 3])
y = tf.nn.relu(tf.matmul(X, W) + b)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4, 0.2, 0.4]])
    (_b, _W, _X, _y) = sess.run((b, W, X, y), feed_dict = {X: X_array})
    print('b: ', _b)
    print('W: ', _W)
    print('X: ', _X)
    print('y: ', _y)

# 创建layer函数以矩阵运算仿真神经网络
def layer(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs

X = tf.placeholder("float", [None, 4])
h = layer(output_dim=3, input_dim=4, inputs=X, activation=tf.nn.relu)
y = layer(output_dim=2, input_dim=3, inputs=h)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4, 0.2, 0.4, 0.5]])
    (layer_X, layer_h, layer_y) = sess.run((X, h, y), feed_dict = {X: X_array})
    print('input Layer X: ', layer_X)
    print('hidden Layer h: ', layer_h)
    print('output Layer y: ', layer_y)
'''
# 建立layer_debug函数显示权重与偏差
def layer_debug(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs, W, b

X = tf.placeholder("float", [None, 4])
h, W1, b1 = layer_debug(output_dim=3, input_dim=4, inputs=X, activation=tf.nn.relu)
y, W2, b2 = layer_debug(output_dim=2, input_dim=3, inputs=h)
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X_array = np.array([[0.4, 0.2, 0.4, 0.5]])
    (layer_X, layer_h, layer_y, W1, b1, W2, b2) = sess.run((X, h, y, W1, b1, W2, b2), feed_dict={X: X_array})
    print('input Layer X: ', layer_X)
    print('W1: ', W1)
    print('b1: ', b1)
    print('hidden Layer h: ', layer_h)
    print('W2: ', W2)
    print('b2: ', b2)
    print('output Layer y: ', layer_y)