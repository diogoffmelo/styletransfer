import tensorflow as tf
from .transform_blocks import conv_layer, conv_tranpose_layer, residual_block


class ResidualAutoEncoder(object):
    """
    inputs: 4d (?, ?, ?, 3) tensorflow tensor in [0, 1]
    output: 4d (?, ?, ?, 3) tensorflow tensor in [0, 1]
    """

    def __init__(self, inputs):
        self._build_network(inputs)

    def _build_network(self, inputs):
        self.input = inputs

        self.conv1 = conv_layer(self.input, 32, 9, 1)
        self.conv2 = conv_layer(self.conv1, 64, 3, 2)
        self.conv3 = conv_layer(self.conv2, 128, 3, 2)

        self.resid1 = residual_block(self.conv3, 3)
        self.resid2 = residual_block(self.resid1, 3)
        self.resid3 = residual_block(self.resid2, 3)
        self.resid4 = residual_block(self.resid3, 3)
        self.resid5 = residual_block(self.resid4, 3)

        self.conv_t1 = conv_tranpose_layer(self.resid5, 64, 3, 2)
        self.conv_t2 = conv_tranpose_layer(self.conv_t1, 32, 3, 2)
        self.conv_t3 = conv_layer(self.conv_t2, 3, 9, 1, actv=tf.nn.tanh)

        self.conv_t3n = 0.5 * (self.conv_t3 + 1)
        self._output = self.input + self.conv_t3
        self.output = tf.clip_by_value(self._output, 0, 1.0)
