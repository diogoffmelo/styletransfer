# based on
# https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md
# https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py
# https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py
import tensorflow as tf
import numpy as np
from scipy.io import loadmat as load_from_matlab

STRIDE1 = (1, 1, 1, 1)
PIXEL_SCALE = float(255)
PIXEL_OFFSET = np.array([103.939, 116.779, 123.68])


def conv(inputs, matlayer):
    # Raise an error before is too late ;)
    W, B = matlayer['weights'][0]
    It, Ih, Iw, Ic = inputs.shape
    Ww, Wh, Wi, Wo = W.shape
    assert Wi == Ic
    assert B.shape == (1, Wo)

    tfW = tf.constant(W)
    tfB = tf.constant(B)
    return tf.nn.conv2d(inputs, tfW, STRIDE1, 'SAME') + tfB


def relu(inputs, matlayer):
    # Raise an error before is too late.
    assert matlayer['type'][0] == 'relu'

    return tf.nn.relu(inputs)


def maxpool(inputs, matlayer):
    # Yep, another error check.
    assert matlayer['type'][0] == 'pool'

    return tf.layers.max_pooling2d(inputs, 2, 2, padding='VALID')


def flatten(inputs, matlayer):
    # Better sooner than late.
    W, B = matlayer['weights'][0]
    It, Ih, Iw, Ic = inputs.shape
    Ww, Wh, Wi, Wo = W.shape
    assert (Wh, Ww, Wi) == (Ih, Iw, Ic)
    assert B.shape == (1, Wo)
    
    return tf.layers.flatten(inputs)


def dense(inputs, matlayer):
    # Yo know what they say about precautoins.
    W, B = matlayer['weights'][0]
    Wh, Ww, Wi, Wo = W.shape
    It, Il = inputs.shape
    Wri, Wro = W.reshape([-1, Wo]).shape
    assert (Wh * Ww * Wi) == Wri
    assert Wro == Wo    
    assert B.shape == (1, Wo)
    assert Wri == Il
    
    tfW = tf.constant(W.reshape([-1, Wo]))
    tfB = tf.constant(B)
    return tf.matmul(inputs, tfW) + tfB


def softmax(inputs, matlayer):
    # I know, i'm getting old, so what?
    assert matlayer['type'][0] == 'softmax'
    
    return tf.nn.softmax(inputs)


def layerTypeName(matlayer):
    ltype, lname = matlayer['type'][0], matlayer['name'][0]
    
    # Optimizes fully connected layers as tensorflow dense layer
    if ltype == 'conv' and lname.startswith('fc'):
        ltype = 'dense'
    
    return ltype, lname


LAYERS_LOADER = {
    'conv': conv,
    'relu': relu,
    'pool': maxpool,
    'dense': dense,
    'softmax': softmax,
}


class VGG19():
    """
    inputs: 4d (?, 244, 244, 3) tensorflow tensor in [0, 1]
    """

    def __init__(self, vgg19path, inputs, lastlayer='prob', verbose=True):
        matvgg19 = load_from_matlab(vgg19path)
        matclasses = matvgg19['classes'][0, 0]
        self.matlayers = matvgg19['layers'][0]
        self._build_classes(matclasses)
        self._build_network(inputs, lastlayer=lastlayer, verbose=verbose)

    def _build_classes(self, matclasses):
        idxs, names = matclasses
        self.idxs = idxs[0]
        self.names = names[0]
        self.unique_names = np.asarray([n[0].split(',')[0] for n in self.names])

    def probs_to_names(self, probs, unique_name=True, top=5, include_prob=True):
        most_prob = []
        names = self.unique_names if unique_name else self.names
        if probs.ndim == 1:
            probs = np.expand_dims(probs, axis=0)

        idxs_matrix = np.argsort(probs)[:, -top:]
        for i in range(probs.shape[0]):
            idxs = idxs_matrix[i, ::-1]

            inames = list(names[idxs])
            iprobs = list(probs[i, idxs])

            if include_prob:
                most_prob.append(list(zip(inames, iprobs)))
            else:
                most_prob.append(inames)

        return most_prob


    def _build_network(self, inputs, lastlayer='prob', verbose=True):
        """
        inputs: 4d (Batch, 244, 244, 3) tensorflow tensor in [0, 1]
        """

        self.input = inputs
        self.processed_input = PIXEL_SCALE * self.input - PIXEL_OFFSET

        x = self.processed_input
        for matlayer in self.matlayers:
            matlayer = matlayer[0, 0]
            ltype, lname = layerTypeName(matlayer)    
            if lname == 'fc6':
                if verbose:
                    print('Creating layer {} of type {}'.format('flatten', 'reshape'))

                x = flatten(x, matlayer)
                setattr(self, 'flatten', x)

            if verbose:
                print('Reading layer: {} of type: {}'.format(lname, ltype))
                    
            x = LAYERS_LOADER[ltype](x, matlayer)
            setattr(self, lname, x)

            if lname == lastlayer:
                break

        self.output = x
