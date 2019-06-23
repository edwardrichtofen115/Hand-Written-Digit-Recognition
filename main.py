import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot = True)

import numpy as np
from matplotlib import pyplot as plt
first_image = mnist.train.images[0]
first_image = np.array(first_image,dtype='float')
first_image = first_image.reshape((28,28))
plt.imshow(first_image)
plt.show()

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

weights = {
    'h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
}

biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}


def forward_propagation(x,weights,biases):
    in_layer1 = tf.add(tf.matmul(x,weights['h1']),biases['h1'])
    out_layer1 = tf.nn.relu(in_layer1)
    
    in_layer2 = tf.add(tf.matmul(out_layer1,weights['h2']),biases['h2'])
    out_layer2 = tf.nn.relu(in_layer2)
    
    output = tf.add(tf.matmul(out_layer2,weights['out']),biases['out'])
    ##out_layer3 = tf.nn.relu(in_layer3)
    return output
    
    
x = tf.placeholder("float")
y = tf.placeholder(tf.int32)
pred = forward_propagation(x,weights,biases)

predictions = tf.argmax(pred,axis=1) ## because the pred calculated is one hot encoded , we need to find the index with max value for the prediction, we do it using argamx 
true_labels = tf.argmax(y,axis=1)
correct_predictions = tf.equal(predictions,true_labels)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.04)
optimize = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(50):
    c,_ = sess.run([cost,optimize],feed_dict={x:mnist.train.images,y:mnist.train.labels})
    print(c)

predictions = tf.argmax(pred,axis=1) ## because the pred calculated is one hot encoded , we need to find the index with max value for the prediction, we do it using argamx 
true_labels = tf.argmax(y,axis=1)
correct_predictions = tf.equal(predictions,true_labels)

final_predictions,labels,correct_pred = sess.run([predictions,true_labels,correct_predictions],feed_dict = {x:mnist.test.images,y:mnist.test.labels})
##final_predictions,labels,correct_pred
correct_pred.sum()
