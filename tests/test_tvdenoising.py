import repackage
repackage.up()

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
from skimage.transform import resize as imresize
from src.autoencoder import ResidualAutoEncoder
from src.tools import imread, imbatch1
from src.style import _tv_loss


axeslist = None
def plot_img_cmp(X, Xt, nimages=3):
    imgidxs = np.random.choice(X.shape[0], nimages)

    global axeslist
    if not np.any(axeslist):
        fig, axeslist = plt.subplots(ncols=nimages, nrows=2, sharex=True, sharey=True)
        for i in range(nimages):
            axeslist[0, i].set_axis_off()
            axeslist[1, i].set_axis_off()

        plt.ion()
        plt.show()

    for i in range(nimages):
        axeslist[0, i].imshow(X[imgidxs[i]])
        axeslist[1, i].imshow(Xt[imgidxs[i]])
        


def get_random_images(N=10):
    def get_image():
        img = imread(np.random.choice(image_paths, 1)[0], H=244)
        while len(img.shape) != 3:
            print('Wrong image shape:', img.shape)
            img = imread(np.random.choice(image_paths, 1)[0], H=244)

        return img

    return np.random.random([N, 244, 244, 3])


tf.reset_default_graph()

inputs = tf.layers.Input(shape=(244, 244, 3), dtype=tf.float32)

autoencoder = ResidualAutoEncoder(inputs)
#output = 0.5*(autoencoder.conv_t3 + 1)
#_output = output
output = autoencoder.conv_t3n
_output = output

#_output = autoencoder._output

with tf.name_scope('loss'):
    loss = _tv_loss(output)


#train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
train = tf.train.AdamOptimizer(0.001).minimize(loss)





sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
ll = []


npimages = get_random_images(7)
with sess.as_default():
    for i in range(101):
        food = {inputs: npimages}
        
        l, _ = sess.run([loss, train], feed_dict=food)
        print('loss @t={}:{}'.format(i, l))

        ll.append(l)
        if i % 2 == 0 and i > 0:
            imgt = sess.run(output, feed_dict=food)
            plot_img_cmp(npimages, imgt, 3)
            plt.draw()
            plt.pause(0.001)






