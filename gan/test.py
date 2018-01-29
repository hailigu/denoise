import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
_data_dir = './asset/data/mnist'
data_set = input_data.read_data_sets(_data_dir, reshape=False, one_hot=False)
print np.shape(data_set.train.images[0])