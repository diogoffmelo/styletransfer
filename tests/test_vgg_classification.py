import repackage
repackage.up()

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from src.vgg import VGG19
from src.tools import imread2vgg, imbatch1

vgg19path = '/home/diogo/data/deep_studies/style_transfer/imagenet-vgg-verydeep-19.mat'

tusker_path = '/home/diogo/data/deep_studies/tusker.jpeg'
tiger_path = '/home/diogo/data/deep_studies/style_transfer/styletransfer/resources/tiger.jpeg'
pluzer_path = '/home/diogo/data/deep_studies/style_transfer/styletransfer/resources/pluzze.jpeg'

img_paths = [tusker_path, tiger_path, pluzer_path] 


def plot_img_cmp(X, probs, ncols=3):
    nrows=2

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for i in range(ncols):
        axeslist[0, i].set_axis_off()

    plt.tight_layout(pad=.8, h_pad=0.05, w_pad=.7, rect=(0, 0.25, 1, 1))
    for i in range(ncols):
        names, scores = zip(*probs[i])
        names = [n.replace(' ', '\n') for n in names]

        #axeslist[0, i].cla()
        axeslist[0, i].imshow(X[i])
        axeslist[0, i].set_axis_off()

        rrange = np.arange(len(scores))

        axeslist[1, i].cla()
        axeslist[1, i].bar(rrange, scores)
        axeslist[1, i].set_xticks(rrange, minor=False)
        axeslist[1, i].set_xticklabels(names, fontdict=None, minor=False, rotation=90)

    plt.ioff()
    plt.show()

tusker_img = imread2vgg('/home/diogo/data/deep_studies/tusker.jpeg')
tiger_img = imread2vgg('/home/diogo/data/deep_studies/style_transfer/styletransfer/resources/tiger.jpeg')
pluzer_img = imread2vgg('/home/diogo/data/deep_studies/style_transfer/styletransfer/resources/pluzze.jpeg')


npimages = np.concatenate([imbatch1(imread2vgg(p)) for p in img_paths])


tf.reset_default_graph()
inputs = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
vgg19 = VGG19(vgg19path, inputs, lastlayer='prob')

sess = tf.Session()
with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    probs = sess.run(vgg19.prob, feed_dict={inputs:npimages})

    plot_img_cmp(npimages, vgg19.probs_to_names(probs), ncols=3)
