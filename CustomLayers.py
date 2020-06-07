"""
Created on Sat May 16 00:25:28 2020

@author: smoss

This is a gan inspired by Big GAN and based on the implementation here https://github.com/taki0112/BigGAN-Tensorflow
"""
import Utils
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.python.keras.utils import tf_utils

weight_init = TruncatedNormal(mean=0.0, stddev=0.02)
weight_regularizer = Utils.orthogonal_regularizer(0.0001)
weight_regularizer_fully = Utils.orthogonal_regularizer_fully(0.0001)

from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow.python.keras.engine.input_spec import InputSpec
import traceback
import math

class Conv2D(layers.Conv2D):

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined, found None')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )

        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=initializers.RandomNormal(0, 1),
            name='sn',
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

        self.input_spec = InputSpec(
            ndim=self.rank + 2,
            axes={channel_axis: input_dim}
        )
        self.built = True

    def call(self, inputs, training=None):
        training_value = tf_utils.constant_value(training)
        def _l2normalize(v):
            return v / (K.sum(v ** 2) ** .5 + 1e-4)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        W_shape = self.kernel.shape.as_list()
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)

        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        W_bar = W_reshaped / sigma

        if not training_value:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        outputs = K.conv2d(
            inputs,
            W_bar,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format
            )
        return outputs

class Dense(layers.Dense):

    def build(self, input_shape):
        input_shape = input_shape.as_list()
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units),
            initializer=self.kernel_initializer,
            name='kernel',
            dtype=K.floatx(),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                name='bias',
                dtype=K.floatx(),
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint
            )

        self.u = self.add_weight(
            shape=tuple([1, self.kernel.shape.as_list()[-1]]),
            initializer=initializers.RandomNormal(0, 1),
            name='sn',
            dtype=K.floatx(),
            trainable=False,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        )

        self.input_spec = InputSpec(
            ndim=2,
            axes={-1: input_dim}
        )
        self.built = True

    def call(self, inputs, training=None):
        training_value = tf_utils.constant_value(training)
        def _l2normalize(v):
            return v / (K.sum(v ** 2) ** .5 + 1e-4)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v

        W_shape = self.kernel.shape.as_list()
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)

        sigma = K.dot(_v, W_reshaped)
        sigma = K.dot(sigma, K.transpose(_u))
        W_bar = W_reshaped / sigma

        if not training_value:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = K.reshape(W_bar, W_shape)

        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
          output = self.activation(output)
        return output

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
        self.gamma = self.add_weight(shape=[1], name='gamma', initializer=tf.constant_initializer(0.0))
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
        self.ReLU = layers.ReLU()
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
        output_t = self.ReLU(output_t)
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
            momentum=.9,
            name=None,
            up=False,
            down=False,
            normalize=True,
            **kwargs
    ):
        super(ConvBlockCond, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel = kernel
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.momentum = momentum
        self.normalize = normalize
        if normalize:
            self.batchNorm = CondBatchNorm()
        self.ReLU = layers.ReLU()
        self.up = up
        self.down = down
        if self.up:
            self.upsample = layers.UpSampling2D()
        elif self.down:
            self.downsample = layers.MaxPool2D()
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
        if self.normalize:
            output_t = self.batchNorm([conv_t, noise_t])
        output_t = self.ReLU(output_t)
        if self.up:
            output_t = self.upsample(output_t)
        elif self.down:
            output_t = self.downsample(output_t)
        output_t = self.conv(output_t)
        return output_t

    def get_config(self):
        config = super(ConvBlockCond, self).get_config()
        config.update()
        return config

    def compute_output_shape(self, input_shape):
        conv_out = input_shape[0]
        img_size = conv_out[1, 2]
        if self.up:
            img_size /= 2
        elif self.down:
            img_size *= 2
        return conv_out[0] + img_size + (self.channels,)

class ConvBlockCondDown(layers.Layer):
    def __init__(
            self,
            channels,
            use_bias=True,
            kernel=3,
            stride=2,
            padding='same',
            momentum=.9,
            name=None,
            up=False,
            down=False,
            normalize=True,
            **kwargs
    ):
        super(ConvBlockCondDown, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.kernel = kernel
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.momentum = momentum
        self.normalize = normalize
        self.ReLU = layers.ReLU()
        self.up = up
        self.down = down
        if self.up:
            self.upsample = layers.UpSampling2D()
        elif self.down:
            self.downsample = layers.MaxPool2D()
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
        output_t = self.ReLU(inputs)
        if self.up:
            output_t = self.upsample(output_t)
        elif self.down:
            output_t = self.downsample(output_t)
        output_t = self.conv(output_t)
        return output_t

    def get_config(self):
        config = super(ConvBlockCond, self).get_config()
        config.update()
        return config

    def compute_output_shape(self, input_shape):
        conv_out = input_shape[0]
        img_size = conv_out[1, 2]
        if self.up:
            img_size /= 2
        elif self.down:
            img_size *= 2
        return conv_out[0] + img_size + (self.channels,)

class ResBlockCondD(layers.Layer):
    def __init__(
            self,
            channelsOut,
            use_bias=True,
            momentum=.9,
            name=None,
            split=True,
            up=False,
            **kwargs
    ):
        super(ResBlockCondD, self).__init__(name=name, **kwargs)
        self.channels = channelsOut
        self.use_bias = use_bias
        self.momentum = momentum
        self.split = split
        self.up = up
        if up:
            self.upsample = layers.UpSampling2D(size=2)

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('ResBlockCondD layer requires a list of inputs')
        conv_in = input_shape[0]
        self.deconv_1 = ConvBlockCond(conv_in[-1] // 4, self.use_bias, kernel=1, momentum=self.momentum, stride=1)
        self.deconv_2 = ConvBlockCond(conv_in[-1] // 4, self.use_bias, momentum=self.momentum, stride=1, up=self.up)
        self.deconv_3 = ConvBlockCond(conv_in[-1] // 4, self.use_bias, momentum=self.momentum, stride=1)
        self.deconv_4 = ConvBlockCond(self.channels, self.use_bias, kernel=1, momentum=self.momentum, stride=1)

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
            skip_t = self.upsample(skip_t)

        return output_t + skip_t

    def get_config(self):
        config = super(ResBlockCondD, self).get_config()
        config.update({
            'channels': self.channels,
            'use_bias': self.use_bias,
            'momentum': self.momentum
        })
        return config

    def compute_output_shape(self, input_shape):
        conv_out = input_shape[0]
        img_size = conv_out[1, 2]
        if self.up:
            img_size /= 2
        return conv_out[0] + img_size + (self.channels,)

class ResBlockCondDownD(layers.Layer):
    def __init__(
            self,
            channelsOut,
            use_bias=True,
            momentum=.9,
            name=None,
            down=False,
            **kwargs
    ):
        super(ResBlockCondDownD, self).__init__(name=name, **kwargs)
        self.channels = channelsOut
        self.use_bias = use_bias
        self.momentum = momentum
        self.concat = layers.Concatenate()
        self.down = down
        if down:
            self.downsample = layers.MaxPool2D()

    def build(self, input_shape):
        channels_in = input_shape[-1]
        self.deconv_1 = ConvBlockCondDown(channels_in // 4, self.use_bias, kernel=1, momentum=self.momentum, stride=1)
        self.deconv_2 = ConvBlockCondDown(channels_in // 4, self.use_bias, momentum=self.momentum, stride=1)
        self.deconv_3 = ConvBlockCondDown(channels_in // 4, self.use_bias, momentum=self.momentum, stride=1)
        self.deconv_4 = ConvBlockCondDown(self.channels, self.use_bias, kernel=1, momentum=self.momentum, stride=1, down=self.down)
        self.extra_conv = tf_utils.constant_value(channels_in < self.channels)
        if self.extra_conv:
            self.conv_extra = Conv2D(self.channels - channels_in, 1, use_bias=self.use_bias)

    def call(self, inputs):
        conv_t = inputs
        output_t = self.deconv_1(conv_t)
        output_t = self.deconv_2(output_t)
        output_t = self.deconv_3(output_t)
        output_t = self.deconv_4(output_t)
        skip_t = conv_t
        if self.down:
            skip_t = self.downsample(skip_t)
        if self.extra_conv:
            extra_t = self.conv_extra(skip_t)
            skip_t = self.concat([extra_t, skip_t])

        return tf.add(output_t, skip_t)

    def get_config(self):
        config = super(ResBlockCondDownD, self).get_config()
        config.update({
            'channels': self.channels,
            'use_bias': self.use_bias,
            'momentum': self.momentum
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0]

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
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('CondBatchNorm layer requires a list of inputs')
        _, _, _, c = list(input_shape[0])

        self.beta = layers.Dense(
            c,
            kernel_initializer=weight_init,
            kernel_regularizer=weight_regularizer_fully,
            use_bias=True,
            name="beta"
        )
        self.gamma = layers.Dense(
            c,
            kernel_initializer=weight_init,
            kernel_regularizer=weight_regularizer_fully,
            use_bias=True,
            name="gamma"
        )
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
            test_mean = batch_mean * (1 - self.momentum) + (self.moving_mean * self.momentum)
            test_var = batch_var * (1 - self.momentum) + (self.moving_variance * self.momentum)
            with tf.control_dependencies([
                self.moving_mean.assign(test_mean),
                self.moving_variance.assign(test_var)
            ]):
                return tf.nn.batch_normalization(
                    norm_t,
                    batch_mean,
                    batch_var,
                    beta_t,
                    gamma_t,
                    self.epsilon
                )
        else:
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

class GlobalSumPooling2D(layers.Layer):
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=(1, 2))
