"""
Created on Sat May 16 00:25:28 2020

@author: smoss

This is a gan inspired by Big GAN and based on the implementation here https://github.com/taki0112/BigGAN-Tensorflow
"""
#TODO Implement spectral normalization
#TODO Update softmax attention block to match original implementation
import Utils
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.python.keras.utils import tf_utils

weight_init = TruncatedNormal(mean=0.0, stddev=0.02)
weight_regularizer = Utils.orthogonal_regularizer(0.0001)
weight_regularizer_fully = Utils.orthogonal_regularizer_fully(0.0001)

from tensorflow.keras.layers import Conv2D as Conv2D
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

        self.theta = Conv2D(self.channels_fg, 1, 1, name='theta')
        self.phi = Conv2D(self.channels_fg, 1, 1, name='phi')
        self.phi_p = layers.MaxPool2D(name='phi2')
        self.g = Conv2D(self.channels_h, 1, 1, name='G')
        self.g_p = layers.MaxPool2D(name='G2')

        self.s = MatMul(transpose_b=True, name='FirstMatMul')
        self.beta = layers.Softmax(name='beta')
        self.o = MatMul(transpose_a=True, name='SecondMatMul')
        self.gamma = self.add_weight(shape=[1], initializer=tf.constant_initializer(0.0))
        self.o3 = Conv2D(self.channels, 1, 1)

    def build(self, input_shape):
        self.hw = input_shape[1] * input_shape[2]
        self.hw_div = math.ceil(input_shape[1] / 2) * math.ceil(input_shape[2] / 2)
        self.hw_flatten_theta = layers.Reshape((self.hw, self.channels_fg), name='reshapeTHETA')
        self.hw_flatten_phi = layers.Reshape((self.hw_div, self.channels_fg), name='reshapePHI')
        self.hw_flatten_g = layers.Reshape((self.hw_div, self.channels_h), name='reshapeG')
        self.o2 = layers.Reshape((input_shape[1], input_shape[2], self.channels_h), name='finalReshape')

    def call(self, inputs):
        theta_t = self.theta(inputs)
        phi_t = self.phi(inputs)
        phi_t = self.phi_p(phi_t)
        g_t = self.g(inputs)
        g_t = self.g_p(g_t)

        theta_t = self.hw_flatten_theta(theta_t)
        phi_t = self.hw_flatten_phi(phi_t)
        s_t = self.s([phi_t, theta_t])
        beta_t = self.beta(s_t)

        g_t = self.hw_flatten_g(g_t)

        o_t = self.o([beta_t, g_t])
        o_t = self.o2(o_t)
        o_t = self.o3(o_t)
        return self.gamma * o_t + inputs

    def get_config(self):
        config = super(SoftAttentionMax, self).get_config()
        config.update({
            'channels': self.channels
        })
        return config

class ConvBlock(layers.Layer):
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
        super(ConvBlock, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel = kernel
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.momentum = momentum
        self.batchNorm = layers.BatchNormalization(momentum=momentum)
        self.leakyRelu = layers.LeakyReLU()
        self.conv = Conv2D(
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
        config = super(ConvBlock, self).get_config()
        config.update()
        return config

class ConvBlockCond(layers.Layer):
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
            up=False,
            **kwargs
    ):
        super(ConvBlockCond, self).__init__(name=None, **kwargs)
        self.channels = channels
        self.kernel = kernel
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.momentum = momentum
        self.batchNorm = CondBatchNorm()
        self.leakyRelu = layers.LeakyReLU()
        self.up = up
        if self.up:
            self.upsample = layers.UpSampling2D()
        self.conv = Conv2D(
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
        if self.up:
            output_t = self.upsample(output_t)
        output_t = self.conv(output_t)
        # print(output_t)
        return output_t

    def get_config(self):
        config = super(ConvBlockCond, self).get_config()
        config.update()
        return config

class ResBlockCondD(layers.Layer):
    def __init__(
            self,
            channelsOut,
            use_bias=True,
            sn=False,
            momentum=.9,
            name=None,
            split=True,
            up=False,
            **kwargs
    ):
        super(ResBlockCondD, self).__init__(name=None, **kwargs)
        self.channels = channelsOut
        self.use_bias = use_bias
        self.momentum = momentum
        self.deconv_1 = ConvBlockCond(channelsOut // 2, use_bias, kernel=1, momentum=momentum, stride=1)
        self.deconv_2 = ConvBlockCond(channelsOut // 2, use_bias, momentum=momentum, stride=1, up=up)
        self.deconv_3 = ConvBlockCond(channelsOut // 2, use_bias, momentum=momentum, stride=1)
        self.deconv_4 = ConvBlockCond(channelsOut, use_bias, kernel=1, momentum=momentum, stride=1)
        self.split = split
        self.up = up
        if up:
            self.Upsample = layers.UpSampling2D(size=2)

    def call(self, inputs):
        conv_t, noise_t = inputs
        output_t = self.deconv_1([conv_t, noise_t])
        output_t = self.deconv_2([output_t, noise_t])
        output_t = self.deconv_3([output_t, noise_t])
        output_t = self.deconv_4([output_t, noise_t])
        skip_t = conv_t
        if self.split:
            skip_t, _ = tf.split(skip_t, 2, axis=3)
        if self.up:
            skip_t = self.Upsample(skip_t)

        return output_t + skip_t

    def get_config(self):
        config = super(ResBlockCondD, self).get_config()
        config.update({
            'channels': self.channels,
            'use_bias': self.use_bias,
            'momentum': self.momentum
        })
        return config

# class EmbeddingBlock(layers.Layer):
#     def __init__(self, init_shape=(4, 4, 1), n_classes=1000, noise_dim=128, name=None, **kwargs):
#         super(EmbeddingBlock, self).__init__(name=name, **kwargs)
#         self.init_shape = init_shape
#         self.n_classes = n_classes
#         self.noise_dim = noise_dim
#         self.embedding = layers.Embedding(n_classes, noise_dim)
#         self.dense = layers.Dense(init_shape[0] * init_shape[1])
#         self.reshape = layers.Reshape(init_shape)
#
#     def call(self, inputs):
#         output_t = self.embedding(inputs)
#         output_t = self.dense(output_t)
#         output_t = self.reshape(output_t)
#         return output_t
#
#     def get_config(self):
#         config = super(CondBatchNorm, self).get_config()
#         config.update({
#             'init_shape': self.init_shape,
#             'n_classes': self.n_classes,
#             'noise_dim': self.noise_dim
#         })
#         return config

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
        momentum=.9,
        axis=-1,
        name=None,
        **kwargs
     ):
        super(CondBatchNorm, self).__init__(name=name, **kwargs)
        self.axis = axis
        # self.dynamic = True
        self.epsilon = epsilon
        self.momentum = momentum

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
            self.add_update([
                K.moving_average_update(
                    self.moving_mean,
                    batch_mean,
                    self.momentum
                ),
                K.moving_average_update(
                    self.moving_variance,
                    batch_var,
                    self.momentum
                )
            ])
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
