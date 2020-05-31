"""
Created on Sat May 16 00:25:28 2020

@author: smoss

This is a gan inspired by Big GAN and based on the implementation here https://github.com/taki0112/BigGAN-Tensorflow
"""
#TODO Implement spectral normalization
import Utils
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.python.keras.utils import tf_utils

weight_init = TruncatedNormal(mean=0.0, stddev=0.02)
weight_regularizer = Utils.orthogonal_regularizer(0.0001)
weight_regularizer_fully = Utils.orthogonal_regularizer_fully(0.0001)

from tensorflow.keras.layers import Conv2DTranspose as deConv2D
from tensorflow.keras.layers import Conv2D as conv2D
from tensorflow.keras import backend as K
import traceback
import math

class MatMul(layers.Layer):

    def __init__(self, transpose_a=False, transpose_b=False, name=None, **kwargs):
        super(MatMul, self).__init__(name=name, **kwargs)
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError('MatMul layer requires a list of inputs')
        return tf.matmul(inputs[0], inputs[1], transpose_a=self.transpose_a, transpose_b=self.transpose_b)

    def get_config(self):
        config = super(MatMul, self).get_config()
        config.update({
            'transpose_a': self.transpose_a,
            'transpose_b': self.transpose_b
        })
        return config

class SoftAttentionMax(layers.Layer):
    def __init__(self, channels, sn=False, name=None, **kwargs):
        super(SoftAttentionMax, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.channels_fg = self.channels // 8
        self.channels_h = self.channels // 2

        self.f = conv2D(self.channels_fg, 1, 1, name='F')
        self.f_p = layers.MaxPool2D(name='F2')
        self.g = conv2D(self.channels_fg, 1, 1, name='G')
        self.h = conv2D(self.channels_h, 1, 1, name='H')
        self.h_p = layers.MaxPool2D(name='H2')

        self.s = MatMul(transpose_b=True, name='FirstMatMul')
        self.beta = layers.Softmax(name='beta')
        self.o = MatMul(name='SecondMatMul')
        self.gamma = self.add_weight(shape=[1], initializer=tf.constant_initializer(0.0))
        self.o3 = conv2D(self.channels, 1, 1)

    def build(self, input_shape):
        self.hw = input_shape[1] * input_shape[2]
        self.hw_div = math.ceil(input_shape[1] / 2) * math.ceil(input_shape[2] / 2)
        self.hw_flatten_f = layers.Reshape((self.hw_div, self.channels_fg), name='reshapeF')
        self.hw_flatten_g = layers.Reshape((self.hw, self.channels_fg), name='reshapeG')
        self.hw_flatten_h = layers.Reshape((self.hw_div, self.channels_h), name='reshapeH')
        self.o2 = layers.Reshape((input_shape[1], input_shape[2], self.channels_h), name='finalReshape')

    def call(self, inputs):
        f_t = self.f(inputs)
        f_t = self.f_p(f_t)
        g_t = self.g(inputs)
        h_t = self.h(inputs)
        h_t = self.h_p(h_t)

        f_t = self.hw_flatten_f(f_t)
        g_t = self.hw_flatten_g(g_t)
        s_t = self.s([g_t, f_t])
        beta_t = self.beta(s_t)

        h_t = self.hw_flatten_h(h_t)

        o_t = self.o([beta_t, h_t])
        o_t = self.o2(o_t)
        o_t = self.o3(o_t)
        return self.gamma * o_t + inputs

    def get_config(self):
        config = super(SoftAttentionMax, self).get_config()
        config.update({
            'channels': self.channels
        })
        return config

class DeconvBlock(layers.Layer):
    def __init__(
            self,
            channels,
            use_bias=True,
            kernel=3,
            stride=2,
            padding='same',
            sn=False,
            momentum=.9,
            name=None,
            **kwargs
    ):
        super(DeconvBlock, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel = kernel
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.momentum = momentum
        self.batchNorm = layers.BatchNormalization(momentum=momentum)
        self.leakyRelu = layers.LeakyReLU()
        self.deconv = deConv2D(
            channels,
            kernel,
            stride,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=weight_init,
            kernel_regularizer=weight_regularizer
        )

    def call(self, inputs):
        output_t = self.batchNorm(inputs)
        output_t = self.leakyRelu(output_t)
        output_t = self.deconv(output_t)

        return output_t

    def get_config(self):
        config = super(DeconvBlock, self).get_config()
        config.update()
        return config

class DeconvBlockCond(layers.Layer):
    def __init__(
            self,
            channels,
            use_bias=True,
            kernel=3,
            stride=2,
            padding='same',
            sn=False,
            momentum=.9,
            name=None,
            **kwargs
    ):
        super(DeconvBlockCond, self).__init__(name=None, **kwargs)
        self.channels = channels
        self.kernel = kernel
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.momentum = momentum
        self.batchNorm = CondBatchNorm()
        self.leakyRelu = layers.LeakyReLU()
        self.deconv = deConv2D(
            channels,
            kernel,
            stride,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=weight_init,
            kernel_regularizer=weight_regularizer
        )

    def call(self, inputs):
        conv_t, noise_t = inputs
        output_t = self.batchNorm([conv_t, noise_t])
        output_t = self.leakyRelu(output_t)
        output_t = self.deconv(output_t)
        # print(output_t)
        return output_t

    def get_config(self):
        config = super(DeconvBlock, self).get_config()
        config.update()
        return config

class ResBlockUp(layers.Layer):
    def __init__(self, channels, use_bias=True, sn=False, momentum=.9, name=None, **kwargs):
        super(ResBlockUp, self).__init__(name=None, **kwargs)
        self.channels = channels
        self.use_bias = use_bias
        self.momentum = momentum
        self.deconv_1 = DeconvBlock(channels, use_bias, momentum=momentum)
        self.deconv_2 = DeconvBlock(channels, use_bias, momentum=momentum, stride=1)
        self.deconv_3 = deConv2D(channels, 3, 2, use_bias=use_bias, padding='same')

    def call(self, inputs):
        output_t = self.deconv_1(inputs)
        output_t = self.deconv_2(output_t)
        skip_t = self.deconv_3(inputs)

        return output_t + skip_t

    def get_config(self):
        config = super(ResBlockUp, self).get_config()
        config.update({
            'channels': self.channels,
            'use_bias': self.use_bias,
            'momentum': self.momentum
        })
        return config

class ResBlockUpCond(layers.Layer):
    def __init__(self, channels, use_bias=True, sn=False, momentum=.9, name=None, **kwargs):
        super(ResBlockUpCond, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.use_bias = use_bias
        self.momentum =momentum
        self.deconv_1 = DeconvBlockCond(channels, use_bias, momentum=momentum, name='Deconv_block_1')
        self.deconv_2 = DeconvBlockCond(channels, use_bias, momentum=momentum, stride=1, name='Deconv_block_2')
        self.deconv_3 = deConv2D(channels, 3, 2, use_bias=use_bias, padding='same', name='Deconv_skip')

    def call(self, inputs):
        conv_t, noise_t = inputs
        output_t = self.deconv_1([conv_t, noise_t])
        output_t = self.deconv_2([output_t, noise_t])
        skip_t = self.deconv_3(conv_t)

        return output_t + skip_t

    def get_config(self):
        config = super(ResBlockUp, self).get_config()
        config.update({
            'channels': self.channels,
            'use_bias': self.use_bias,
            'momentum': self.momentum
        })
        return config

class EmbeddingBlock(layers.Layer):
    def __init__(self, init_shape=(4, 4, 1), n_classes=1000, noise_dim=1280, name=None, **kwargs):
        super(EmbeddingBlock, self).__init__(name=name, **kwargs)
        self.init_shape = init_shape
        self.n_classes = n_classes
        self.noise_dim = noise_dim
        self.embedding = layers.Embedding(n_classes, noise_dim)
        self.dense = layers.Dense(init_shape[0] * init_shape[1])
        self.reshape = layers.Reshape(init_shape)

    def call(self, inputs):
        output_t = self.embedding(inputs)
        output_t = self.dense(output_t)
        output_t = self.reshape(output_t)
        return output_t

    def get_config(self):
        config = super(CondBatchNorm, self).get_config()
        config.update({
            'init_shape': self.init_shape,
            'n_classes': self.n_classes,
            'noise_dim': self.noise_dim
        })
        return config

class SplitLayer(layers.Layer):
    def __init__(self, split_list, name=None, **kwargs):
        super(SplitLayer, self).__init__(name=name, **kwargs)
        self.split_list = split_list

    def call(self, inputs):
        return tf.split(inputs, self.split_list, -1)

    def get_config(self):
        config = super(CondBatchNorm, self).get_config()
        config .update({
            'split_list': self.split_list
        })
        return config

class CondBatchNorm(layers.BatchNormalization):
    def __init__(self,
        epsilon=1e-5,
        decay=.9,
        name=None,
        **kwargs
     ):
        super(CondBatchNorm, self).__init__(name=name, **kwargs)
        # self.dynamic = True
        self.epsilon = epsilon
        self.decay = decay

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('CondBatchNorm layer requires a list of inputs')
        _, _, _, c = list(input_shape[0])

        self.beta = layers.Dense(c, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, name="beta")
        self.gamma = layers.Dense(c, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, name="gamma")
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=[c],
            initializer='zeros',
            trainable=False
        )

        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=[c],
            initializer='ones',
            trainable=False
        )


        self.resphape_beta = layers.Reshape([1, 1, c], name='reshapeGamma')
        self.resphape_gamma = layers.Reshape([1, 1, c], name='reshapeBeta')

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def call(self, inputs, training=None):
        training = self._get_training_value(training)
        training_value = tf_utils.constant_value(training)
        if len(inputs) != 2:
            raise ValueError('CondBatchNorm layer requires a list of inputs')
        norm_t, noise_t = inputs
        beta_t = self.beta(noise_t)
        beta_t = self.resphape_beta(beta_t)

        gamma_t = self.gamma(noise_t)
        gamma_t = self.resphape_gamma(gamma_t)

        if training_value:
            batch_mean, batch_var = tf.nn.moments(norm_t, [0, 1, 2], name='batchMoments')
            self.add_update(
                self._assign_new_value(
                    self.moving_mean,
                    self.moving_mean * self.decay + batch_mean * (1 - self.decay)
                )
            )
            self.add_update(
                self._assign_new_value(
                    self.moving_variance,
                    self.moving_variance * self.decay + batch_var * (1 - self.decay)
                )
            )
            with tf.control_dependencies([self.moving_mean, self.moving_variance]):
                return tf.nn.batch_normalization(
                    norm_t,
                    batch_mean,
                    batch_var,
                    beta_t,
                    gamma_t,
                    self.epsilon
                )
        else:
            # return norm_t
            return tf.nn.batch_normalization(
                norm_t,
                self.moving_mean,
                self.moving_variance,
                beta_t,
                gamma_t,
                self.epsilon
            )


    def get_config(self):
        config = super(CondBatchNorm, self).get_config()
        config .update({
            'decay': self.decay,
            'epsilon': self.epsilon
        })
        return config
