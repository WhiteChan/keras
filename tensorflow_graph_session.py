import tensorflow as tf 
'''
# tensorflow常数
ts_c = tf.constant(2, name='ts_c')
print(ts_c)

# tensorflow变量
ts_x = tf.Variable(ts_c + 5, name='ts_x')
print(ts_x)

# 建立Session
sess = tf.Session()
# 执行TensorFlow来初始化变量
init = tf.global_variables_initializer()
sess.run(init)
print('ts_c = ', sess.run(ts_c))
print('ts_x = ', sess.run(ts_x))
# 另一个执行图的方法
print('ts_c = ', ts_c.eval(session=sess))
print('ts_x = ', ts_x.eval(session=sess))
# 关闭Session
sess.close()

# With语句打开Session并且自动关闭
a = tf.constant(2, name='a')
x = tf.Variable(a + 5, name='x')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('a = ', sess.run(a))
    print('x = ', sess.run(x))

# placeholder
width = tf.placeholder("int32")
height = tf.placeholder("int32")
area = tf.multiply(width, height)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('area = ', sess.run(area, feed_dict = {width: 6, height: 8}))

# TensorBoard
tf.summary.merge_all()
train_writer = tf.summary.FileWriter('log/area', sess.graph)
'''

'''
# 建立一维与二维张量
ts_X = tf.Variable([[0.4, 0.2, 0.4]])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X = sess.run(ts_X)
    print(X)
    print(X.shape)

W = tf.Variable([[-0.5, -0.2], [-0.3, 0.4], [-0.5, 0.2]])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    W_array = sess.run(W)
    print(W_array)
    print(W_array.shape)
'''

# 矩阵基本运算
# 矩阵乘法
X = tf.Variable([[1., 1., 1.]])
W = tf.Variable([[-0.5, -0.2], [-0.3, 0.4], [-0.5, 0.2]])

XW = tf.matmul(X, W)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(XW))
# 矩阵加法
b = tf.Variable([[0.1, 0.2]])
XW = tf.Variable([[-1.3, 0.4]])

Sum = XW + b
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('Sum: ')
    print(sess.run(Sum))
# 乘法与加法
X = tf.Variable([[1., 1., 1.]])
W = tf.Variable([[-0.5, -0.2], [-0.3, 0.4], [-0.5, 0.2]])
b = tf.Variable([[0.1, 0.2]])

XWb = tf.matmul(X, W) + b

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb: ')
    print(sess.run(XWb))