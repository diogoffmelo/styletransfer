import repackage
repackage.up()

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc


from src.vgg import VGG19
from src.tools import imread2vgg, imbatch1

vgg19path = '/home/diogo/data/deep_studies/style_transfer/imagenet-vgg-verydeep-19.mat'

coco_path = '/home/diogo/data/deep_studies/style_transfer/fast-style-transfer/data/train2014/'
image_paths = [coco_path + name for name in os.listdir(coco_path)]


axeslist, fig = None, None
def plot_img_cmp(X, probs, ncols=3, pause=10):
    nrows=2

    global axeslist, fig
    if not np.any(axeslist):
        fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
        for i in range(ncols):
            axeslist[0, i].set_axis_off()

        plt.tight_layout(pad=.8, h_pad=0.05, w_pad=.7, rect=(0, 0.25, 1, 1))

        plt.ion()
        plt.show()

    for i in range(ncols):
        names, scores = zip(*probs[i])
        names = [n.replace(' ', '\n') for n in names]

        axeslist[0, i].imshow(X[i])
        axeslist[0, i].set_axis_off()

        rrange = np.arange(len(scores))

        axeslist[1, i].cla()
        axeslist[1, i].bar(rrange, scores)
        axeslist[1, i].set_xticks(rrange, minor=False)
        axeslist[1, i].set_xticklabels(names, fontdict=None, minor=False, rotation=90)

    plt.draw()
    plt.pause(pause)


def get_random_images(N=10):
    assert N > 1
    def get_image():
        img = None
        while not np.any(img):
            img_path = np.random.choice(image_paths, 1)[0]
            try:
                img = imread2vgg(img_path)
                assert img.shape == (224, 224, 3)
            except e:
                print(e)
                img = None

        return img

    return np.asarray([get_image() for _ in range(N)], dtype=np.float32)

tf.reset_default_graph()

inputs = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32)
vgg19 = VGG19(vgg19path, inputs, lastlayer='prob')

sess = tf.Session()
with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(7):
        npimages = get_random_images(3)
        probs = sess.run(vgg19.prob, feed_dict={inputs:npimages})

        plot_img_cmp(npimages, vgg19.probs_to_names(probs), ncols=3)
