import repackage
repackage.up()

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


from src.vgg import VGG19
from src.tools import imread2vgg, imbatch1
from src.style import (content_features, style_features, _tv_loss,
                       style_losses, content_losses, tv_losses, build_losses)

STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
CONTENT_LAYERS = ['conv4_2']

vgg19path = '/home/diogo/data/deep_studies/style_transfer/imagenet-vgg-verydeep-19.mat'

tiger_img = imread2vgg('/home/diogo/data/deep_studies/style_transfer/styletransfer/resources/tiger.jpeg')
tusker_img = imread2vgg('/home/diogo/data/deep_studies/tusker.jpeg')
night_img = imread2vgg('/home/diogo/data/deep_studies/style_transfer/night.jpeg')

built_image = np.random.random([224, 224, 3])
#content_image = np.zeros([224, 224, 3], dtype=np.float32)

content_image = tiger_img
style_image = night_img


def plot_series(axis, ys, keep=100):
    ys = np.asarray(ys, dtype=np.float32)
    N = ys.shape[0]
    offset = max(0, N-keep)
    xs = np.arange(offset, N)

    gminv, gmaxv, gmeanv = ys.min(), ys.max(), ys.mean()
    text = 'min={:0>8.4f}\nmean={:0>8.4f}\nmax={:0>8.4f}'.format(gminv, gmeanv, gmaxv)

    ys = ys[xs]
    minv, maxv, meanv = ys.min(), ys.max(), ys.mean()

    axis.cla()
    axis.plot(xs, ys, marker='o', linestyle='--', color='b')
    axis.plot(xs[[0, -1]], [minv, minv], 'r:')
    axis.plot(xs[[0, -1]], [maxv, maxv], 'r:')
    axis.plot(xs[[0, -1]], [meanv, meanv], 'r:')
    axis.set_xlabel(text)

axeslist, fig = None, None
def plot_img_cmp(composed_image, closs, sloss, tloss):

    global axeslist, fig
    if not np.any(axeslist):
        fig, axeslist = plt.subplots(ncols=3, nrows=2)
        for i in range(3):
            axeslist[0, i].set_axis_off()

        axeslist[0, 0].set_title('content_image')
        axeslist[0, 1].set_title('composed_image')
        axeslist[0, 2].set_title('style_image')

        # axeslist[1, 0].set_title('content_loss')
        # axeslist[1, 1].set_title('tv_loss')
        # axeslist[1, 2].set_title('style_loss')

        axeslist[0, 0].imshow(content_image)
        axeslist[0, 2].imshow(style_image)


        plt.tight_layout(pad=.8, h_pad=0.2, w_pad=0, rect=(0, 0.1, 1, 1))

        plt.ion()
        plt.show()

    plot_series(axeslist[1, 0], closs)
    plot_series(axeslist[1, 1], tloss)
    plot_series(axeslist[1, 2], sloss)

    axeslist[1, 0].set_title('content_loss')
    axeslist[1, 1].set_title('tv_loss')
    axeslist[1, 2].set_title('style_loss')


    axeslist[0, 1].imshow(composed_image)

    plt.draw()
    plt.pause(0.01)

plot_img_cmp(built_image, [0, 0], [0, 0], [0, 0])

bcontent_image = imbatch1(content_image)
bstyle_image = imbatch1(style_image)
bbuilt_image = imbatch1(built_image) 


tf.reset_default_graph()
inputs = tf.Variable(bbuilt_image, dtype=tf.float32)
vgg19 = VGG19(vgg19path, inputs, lastlayer='relu5_1')

content_feats_op = content_features(vgg19, CONTENT_LAYERS)
style_feats_op = style_features(vgg19, STYLE_LAYERS)

sess = tf.Session()
with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    content_feats_target = sess.run(content_feats_op, feed_dict={inputs: bcontent_image})
    style_feats_target = sess.run(style_feats_op, feed_dict={inputs: bstyle_image})


content_loss_op = build_losses(content_feats_op, content_feats_target, 1e-4)
style_loss_op = build_losses(style_feats_op, style_feats_target, 1e-2)
denoising_loss_op = 10 * _tv_loss(inputs)


mask0 = tf.cast(tf.less(inputs, 0), tf.float32)
mask1 = tf.cast(tf.greater(inputs, 1), dtype=tf.float32)

error01 = 1000*tf.reduce_mean(tf.square(mask0*inputs) + tf.square(mask1*(inputs-1)))


loss = content_loss_op + style_loss_op + denoising_loss_op + error01
#loss = style_loss_op + denoising_loss_op + error01
train = tf.train.AdamOptimizer(0.01).minimize(loss)
#train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
with sess.as_default():
    init = tf.global_variables_initializer()
    sess.run(init)

    closs, sloss, tloss = [], [], []

    while True:
        for i in range(4000):
            _, cl, sl, tl = sess.run([train, content_loss_op, style_loss_op, denoising_loss_op])
            
            closs.append(cl)
            sloss.append(sl)
            tloss.append(tl)
            if (i+1) % 10  == 0:
                built_image = sess.run(inputs)
                print('losses @t={:0>5d}: \tcontent={:0>7.3f}, \tstyle={:0>7.3f}'.format(i+1, cl, sl))
                plot_img_cmp(np.clip(built_image[0], 0, 1.0), closs, sloss, tloss)

        ans = input('Run for another 200 epoches?\n')
        if not (ans[0] in 'sSyY'):
            break
