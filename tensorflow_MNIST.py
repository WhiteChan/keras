import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print('train', mnist.train.num_examples)
# print('validation', mnist.validation.num_examples)
# print('test', mnist.test.num_examples)

# print('train images: ', mnist.train.images.shape)
# print('labels: ', mnist.train.labels.shape)

batch_image_xs, batch_labels_ys = mnist.train.next_batch(batch_size=100)

def layer(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs

x = tf.placeholder("float", [None, 784])
h1 = layer(output_dim=1000, input_dim=784, inputs=x, activation=tf.nn.relu)
h2 = layer(output_dim=800, input_dim=1000, inputs=h1, activation=tf.nn.relu)
h3 = layer(output_dim=256, input_dim=800, inputs=h2, activation=tf.nn.relu)

y_predict = layer(output_dim=10, input_dim=256, inputs=h3, activation=None)
y_label = tf.placeholder("float", [None, 10])

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y_label))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

correct_prediction = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

trainEpochs = 15
batchSize = 100
totalBatchs = int(mnist.train.num_examples / batchSize)
loss_list = []
epoch_list = []
accuracy_list = []
from time import time
startTime = time()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(trainEpochs):
    for i in range(totalBatchs):
        batch_xs, batch_ys = mnist.train.next_batch(batchSize)
        sess.run(optimizer, feed_dict={x: batch_xs, y_label: batch_ys})
    loss, acc = sess.run([loss_function, accuracy], feed_dict={x: mnist.validation.images, y_label: mnist.validation.labels})
    epoch_list.append(epoch)
    loss_list.append(loss)
    accuracy_list.append(accuracy)
    print('Train Epoch: ', epoch + 1, 'Loss = ', loss, 'Accuracy = ', acc)
    print('Train Accuracy', sess.run(accuracy, feed_dict={x: mnist.train.images, y_label: mnist.train.labels}))
    print('Validation Accuracy', acc)
    print("Test Accuracy", sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))

duration = time() - startTime
print('Train Finished takes: ', duration)