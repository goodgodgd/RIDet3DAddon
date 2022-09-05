import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import numpy as np
import math

from utils.util_class import MyExceptionToCatch


class CustomConv2D:
    CALL_COUNT = -1

    def __init__(self, kernel_size=3, strides=1, padding="same", activation="leaky_relu", scope=None, bn=True,
                 bias_initializer=tf.constant_initializer(0.), kernel_initializer=tf.random_normal_initializer(stddev=0.001)):
        # save arguments for Conv2D layer
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope
        self.bn = bn
        self.bias_initializer = bias_initializer
        self.kernel_initializer = kernel_initializer

    def __call__(self, x, filters, name=None):
        CustomConv2D.CALL_COUNT += 1
        index = CustomConv2D.CALL_COUNT
        name = f"conv{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name

        x = layers.Conv2D(filters, self.kernel_size, self.strides, self.padding,
                          use_bias=not self.bn,
                          kernel_regularizer=tf.keras.regularizers.l2(0.001),
                          kernel_initializer=self.kernel_initializer,
                          bias_initializer=self.bias_initializer, name=name
                          )(x)

        if self.bn:
            x = layers.BatchNormalization()(x)

        if self.activation == "leaky_relu":
            x = layers.LeakyReLU(alpha=0.1)(x)
        elif self.activation == "mish":
            x = tfa.activations.mish(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "swish":
            x = tf.nn.swish(x)
        elif self.activation is False:
            x = x
        else:
            raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {self.activation}")

        return x


class CustomSeparableConv2D:
    CALL_COUNT = -1

    def __init__(self, kernel_size=3, strides=1, padding="same", activation="leaky_relu", scope=None, bn=True,
                 **kwargs):
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.scope = scope
        self.bn = bn
        self.kwargs = kwargs

    def __call__(self, x, filters, name=None):
        CustomSeparableConv2D.CALL_COUNT += 1
        index = CustomSeparableConv2D.CALL_COUNT
        name = f"spconv{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name
        x = layers.SeparableConv2D(filters, self.kernel_size, self.strides, self.padding,
                                   use_bias=not self.bn, name=name,
                                   **self.kwargs
                                   )(x)

        if self.bn:
            x = layers.BatchNormalization()(x)

        if self.activation == "leaky_relu":
            x = layers.LeakyReLU(alpha=0.1)(x)
        elif self.activation == "mish":
            x = tfa.activations.mish(x)
        elif self.activation == "relu":
            x = layers.ReLU()(x)
        elif self.activation == "swish":
            x = tf.nn.swish(x)
        elif self.activation is False:
            x = x
        else:
            raise MyExceptionToCatch(f"[backbone_factory] invalid backbone name: {self.activation}")

        return x


class CustomMax2D:
    CALL_COUNT = -1

    def __init__(self, pool_size=3, strides=1, padding="same", scope=None):
        # save arguments for Conv2D layer
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.scope = scope

    def __call__(self, x, name=None):
        CustomMax2D.CALL_COUNT += 1
        index = CustomMax2D.CALL_COUNT
        name = f"maxpool{index:03d}" if name is None else f"{name}/{index:03d}"
        name = f"{self.scope}/{name}" if self.scope else name

        x = layers.MaxPooling2D(self.pool_size, self.strides, self.padding, name=name)(x)
        return x


class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape, dtype=np.float32) * -math.log((1 - self.probability) / self.probability)

        return result
