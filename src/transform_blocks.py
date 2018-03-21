"""
Code adapted from:
    https://github.com/lengstrom/fast-style-transfer

Please contact him for private usage licence.
"""

import tensorflow as tf

WEIGHTS_INIT_STDEV = 0.1
EPSILON = 1e-3


def _identity(x):
    return x


def instance_norm(inputs):
    """
    Applies instance nomalization (a.k.a. contrast normalization)

    link: https://arxiv.org/abs/1607.08022
    """
    with tf.name_scope('instance_norm'):
        T, H, W, C = [i.value for i in inputs.get_shape()]
        
        # The shift and scale will be learned from the examples
        shift = tf.Variable(tf.zeros([C]))
        scale = tf.Variable(tf.ones([C]))
        
        mu, sigma2 = tf.nn.moments(inputs, [1, 2], keep_dims=True)    
        
        # EPSILON is added for stability (when sigma2 << 1)
        normalized = (inputs - mu) / tf.sqrt(sigma2 + EPSILON)
        out = scale * normalized + shift
    
    return out


def conv_layer(inputs, filters, size, stride, actv=tf.nn.relu, stddev=WEIGHTS_INIT_STDEV):
    """
    Creates a same-padding convolutional layer on inputs with optional
    activation instance nomalization.

    Args:
    inputs   --- 4D tensorflow input tensor.
    filters  --- Number of output channels.
    size     --- list-like size of the kernel. Use an integer to repeated size. 
    stride   --- list-like size padding specs. Use an integer to repeated stride.
    actv     --- tensorflow activation function.
    stddev   --- Standard deviation to the truncated normal weights initialization (default=0.1).

    Examples:
    # 3x3x12 conv layer, stride=[1,1], neither batch nor activation 
    >>>conv_layer(inputs, 12, 3, 1)

    #same, but relu activation
    >>>conv_layer(inputs, 12, 3, 1, tf.nn.relu)

    # 5x5x6 conv layer, stride=[2,2], with batch and relu activation 
    >>>conv_layer(inputs, 6, 5, 2, tf.nn.relu, train_placeholder)
    """
    with tf.name_scope('conv_layer'):
        actv = actv if actv else _identity
        conv = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=size,
            strides=stride,
            padding='same',
            use_bias=False,
            kernel_initializer=tf.initializers.truncated_normal(mean=0, stddev=stddev),
        )
        out = actv(instance_norm(conv))

    return out


def conv_tranpose_layer(inputs, filters, size, stride, stddev=WEIGHTS_INIT_STDEV):
    """
    Creates a same-padding transposed convolutional layer on inputs (always) with 
    instance nomalization and relu activation.

    Args:
    inputs   --- 4D tensorflow input tensor.
    filters  --- Number of output channels.
    size     --- list-like size of the kernel. Use an integer to repeated size. 
    stride   --- list-like size padding specs. Use an integer to repeated stride.
    stddev   --- Standard deviation to the truncated normal weights initialization (default=0.1).

    Examples:
    # 3x3x12 transposed conv layer, stride=[1,1] 
    >>>conv_layer(inputs, 12, 3, 1)
    """
    with tf.name_scope('conv_transpose_layer'):
        actv = tf.nn.relu
        convt = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            kernel_size=size,
            strides=stride,
            padding='same',
            activation=None,
            use_bias=False,
            kernel_initializer=tf.initializers.truncated_normal(mean=0, stddev=stddev),
        )
        out = actv(instance_norm(convt))
    
    return out


def residual_block(inputs, size=3):
    with tf.name_scope('residual_block'):
        T, H, W, C = [i.value for i in inputs.get_shape()]
        
        finputs = conv_layer(inputs, C, size, 1, tf.nn.relu)
        finputs = conv_layer(finputs, C, size, 1, None)
        out = inputs + finputs

    return out
