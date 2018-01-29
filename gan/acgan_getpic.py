# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import logging
from mnist import Mnist
import tensorflow.contrib.slim as slim
from generator import generator
from discriminator import discriminator
from optimizer import optim
from queue_context import queue_context
import os
import sys


_logger = tf.logging._logger
_logger.setLevel(0)


#
# hyper parameters
#

batch_size = 32   # batch size
cat_dim = 10  # total categorical factor
con_dim = 2  # total continuous factor
rand_dim = 38
num_epochs = 30
debug_max_steps = 1000
save_epoch = 5
max_epochs = 50

#
# inputs
#

# MNIST input tensor ( with QueueRunner )
data = Mnist(batch_size=batch_size, num_epochs=num_epochs)
num_batch_per_epoch = data.train.num_batch


# input images and labels
x = data.train.image
y = data.train.label

# labels for discriminator
y_real = tf.ones(batch_size)
y_fake = tf.zeros(batch_size)


# discriminator labels ( half 1s, half 0s )
y_disc = tf.concat(0, [y, y * 0])
#
# create generator
#

# get random class number
z_cat = tf.multinomial(tf.ones((batch_size, cat_dim), dtype=tf.float32) / cat_dim, 1)
z_cat = tf.squeeze(z_cat, [-1])
z_cat = tf.cast(z_cat, tf.int32)

# continuous latent variable
z_con = tf.random_normal((batch_size, con_dim))
z_rand = tf.random_normal((batch_size, rand_dim))

z = tf.concat(1, [tf.one_hot(z_cat, depth = cat_dim), z_con, z_rand])

# generator network
gen = generator(z)

# add image summary
# tf.sg_summary_image(gen)
tf.image_summary('real', x)
tf.image_summary('fake', gen)
#
# discriminator
disc_real, cat_real, _ = discriminator(x)
disc_fake, cat_fake, con_fake = discriminator(gen)

# discriminator loss
loss_d_r = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, y_real))
loss_d_f = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, y_fake))
loss_d = (loss_d_r + loss_d_f) / 2
# generator loss
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, y_real))

# categorical factor loss
loss_c_r = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(cat_real, y))
loss_c_d = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(cat_fake, z_cat))
loss_c = (loss_c_r + loss_c_d) / 2
# continuous factor loss
loss_con =tf.reduce_mean(tf.square(con_fake-z_con))



train_disc, disc_global_step = optim(loss_d + loss_c + loss_con, lr=0.0001, optim = 'Adm', category='discriminator')
train_gen, gen_global_step = optim(loss_g + loss_c + loss_con, lr=0.001, optim = 'Adm', category='generator')
init = tf.initialize_all_variables()
saver = tf.train.Saver()
cur_epoch = 0
cur_step = 0
merge = tf.merge_summary(tf.get_collection(tf.GraphKeys.SUMMARIES))
import ipdb;ipdb.set_trace()
with tf.Session() as sess:
    summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)
    sess.run(init)
    saver.restore(sess,'/home/wang/下载/tensorflow-101-master/GAN/AC-GAN/saver-85900')
    summary_str = sess.run(merge)
    summary_writer.add_summary(summary_str, 1)