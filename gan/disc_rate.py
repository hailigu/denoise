import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from discriminator import discriminator
from queue_context import queue_context


_logger = tf.logging._logger
_logger.setLevel(0)

_data_dir = './asset/data/mnist'
batch_size=100
total_correct=0

data_set = input_data.read_data_sets(_data_dir, reshape=False, one_hot=True)
test_x=data_set.test.images
test_y=data_set.test.labels



#import ipdb;ipdb.set_trace()
#print x.get_shape()


x = tf.placeholder("float",shape=[None,28,28,1])
y = tf.placeholder("float",shape=[None,10])

_,y_disc,_=discriminator(x,batch_size=x.get_shape()[0])

#correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_disc,1))
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

with tf.Session() as sess:
    sess.run(tf.group(tf.global_variables_initializer(),tf.local_variables_initializer()))
    saver = tf.train.Saver()
    saver.restore(sess, './saver-85900')  # tf.train.latest_checkpoint('checkpoint_dir')
    for i in range(100):
        batch_xs,batch_ys=data_set.test.next_batch(batch_size)
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_disc,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
        temp=sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys})
        total_correct += temp * 100
print 'accruacy in 10000 test samples is ',total_correct/10000
