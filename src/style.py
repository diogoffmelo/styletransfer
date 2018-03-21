import tensorflow as tf
import numpy as np


def _apply_on_layers(net, layer_names, f):
    return [f(getattr(net, n)) for n in layer_names]


def _style_feature(layer):
    T, H, W, C = [k.value for k in layer.get_shape()]
    reshaped = tf.reshape(layer, [T, H * W, C])        
    return tf.matmul(tf.transpose(reshaped, perm=[0,2,1]), reshaped) / (H * W * H * W)


def content_features(net, layer_names):
    return _apply_on_layers(net, layer_names, lambda x: x)


def style_features(net, layer_names):
    return _apply_on_layers(net, layer_names, _style_feature)


def _tv_loss(layer):
    tvx = tf.abs(layer[:, 1:, :, :] - layer[:, :-1, :, :])
    tvy = tf.abs(layer[:, :, 1:, :] - layer[:, :, :-1, :])
    return tf.reduce_mean(tvx) + tf.reduce_mean(tvy)


def tv_losses(net, layer_names):
    return _apply_on_layers(net, layer_names, _tv_loss)


def _loss(feat, target):
    diff = feat - tf.constant(target)

    # loss l1 would be tf.reduce_mean(tf.abs(diff))
    return tf.reduce_mean(tf.square(diff)) / 2.0


def _losses(net, layer_names, targets_map, feature):
    features = _apply_on_layers(net, layer_names, feature)
    targets = [targets_map[k] for k in layer_names]
    return [_loss(f, t) for f, t in zip(features, targets)]


def style_losses(net, layer_names, targets_map):
    return _losses(net, layer_names, targets_map, _style_feature)


def content_losses(net, layer_names, targets_map):
    return _losses(net, layer_names, targets_map, lambda x: x)


def build_losses(ops, targets, weights=1.0):
    weights = list(weights * np.ones(len(targets), dtype=np.float32))
    return sum([0] + [w*_loss(o, t) for o, t, w in zip(ops, targets, weights)])
