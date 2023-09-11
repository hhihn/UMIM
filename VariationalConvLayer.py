# -*- coding: utf-8 -*-
"""Convolutional MoE layers. The code here is based on the implementation of the standard convolutional layers in Keras.
"""
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras import activations, initializers, regularizers, constraints
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec

import tensorflow_probability as tfp

from tensorflow.python.layers import utils as tf_layers_util  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import nn_ops, nn  # pylint: disable=g-direct-tensorflow-import
from initializer import *

tfd = tfp.distributions


class _VariationalConv(Layer):
    """Abstract nD convolution layer mixture of experts (private, used as implementation base).
    """

    def __init__(self, rank,
                 n_filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format='channels_last',
                 dilation_rate=1,
                 kernel_initializer='he_normal',
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(_VariationalConv, self).__init__(**kwargs)
        self.rank = rank
        self.n_filters = n_filters
        self.n_total_filters = self.n_filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.kernel_initializer = tf.keras.initializers.HeNormal()
        rho_init = -6.0  # self._softplus_inverse(0.006)
        self.rho_initializer = tf.keras.initializers.constant(value=rho_init)
        self.activation = activations.get(activation)

        self.use_bias = use_bias

        self.bias_initializer = initializers.get(bias_initializer)
        self.mu_kernel_regularizer = tf.keras.regularizers.L2(l2=1e-3)
        self.kernel_regularizer = None

        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)

        self.bias_constraint = constraints.get(bias_constraint)

        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
            self.tf_data_format = 'NCHW'
        else:
            channel_axis = -1
            self.tf_data_format = 'NHWC'

        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')

        input_dim = input_shape[channel_axis]

        kernel_shape = self.kernel_size + (input_dim, self.n_filters)
        rho_shape = self.kernel_size[1:] + (input_dim, self.n_filters)

        self.mu_kernel = self.add_weight(shape=kernel_shape,
                                         initializer=self.kernel_initializer,
                                         name='post_components_q_mu_kernels',
                                         regularizer=self.mu_kernel_regularizer,
                                         constraint=self.kernel_constraint)

        self.rho_kernel = self.add_weight(shape=rho_shape,
                                          initializer=self.rho_initializer,
                                          name='post_q_mu_log_var',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)

        self.prior_mu_kernel = self.add_weight(shape=kernel_shape,
                                               initializer="glorot_uniform",
                                               name='prior_components_q_mu_kernels',
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint,
                                               trainable=False)

        self.prior_rho_kernel = self.add_weight(shape=rho_shape,
                                                initializer=self.rho_initializer,
                                                name='prior_q_mu_log_var',
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint,
                                                trainable=False)
        if self.use_bias:
            bias_shape = (self.n_filters,)
            self.bias = self.add_weight(shape=bias_shape,
                                        initializer=self.bias_initializer,
                                        name='post_bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.prior_bias = self.add_weight(shape=bias_shape,
                                              initializer=self.bias_initializer,
                                              name='prior_bias',
                                              regularizer=self.bias_regularizer,
                                              constraint=self.bias_constraint,
                                              trainable=False)
        else:
            self.bias = None
            self.prior_bias = None

        kernel_shape = self.kernel_size + (input_dim, self.n_filters)
        self._convolution_op = nn_ops.Convolution(
            input_shape,
            filter_shape=tf.TensorShape(kernel_shape),
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=self.padding.upper(),
            data_format=tf_layers_util.convert_data_format(
                self.data_format, self.rank + 2))

        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        outputs = self.apply_kernels(inputs)
        #prior_outputs = self.apply_kernels(inputs, prior=True)

        # DKL losses
        self.p_w_d = tfd.MultivariateNormalDiag(
            loc=self.mu_kernel,
            scale_diag=tf.nn.softplus(self.rho_kernel) + 1e-6,
            validate_args=True
        )

        self.prior_p_w_d = tfd.MultivariateNormalDiag(
            loc=self.prior_mu_kernel,
            scale_diag=tf.nn.softplus(self.prior_rho_kernel) + 1e-6,
            validate_args=True
        )

        w_dkl = tfd.kl_divergence(self.p_w_d, self.prior_p_w_d)  # gaussian
        self.add_loss(losses=tf.reduce_sum(w_dkl))

        return outputs#, prior_outputs

    def apply_kernels(self, input_tensors, prior=False):

        def random_rademacher(shape, dtype=tf.float32, seed=None):
            int_dtype = tf.int64 if tf.as_dtype(dtype) != tf.int32 else tf.int32
            random_bernoulli = tf.random.uniform(
                shape, minval=0, maxval=2, dtype=int_dtype, seed=seed)
            return tf.cast(2 * random_bernoulli - 1, dtype)

        if prior:
            mu_kernel = self.prior_mu_kernel
            rho_kernel = self.prior_rho_kernel
            bias = self.prior_bias
        else:
            mu_kernel = self.mu_kernel
            rho_kernel = self.rho_kernel
            bias = self.bias
        input = input_tensors
        input_shape = tf.shape(input)
        batch_shape = tf.expand_dims(input_shape[0], 0)
        if self.data_format == 'channels_first':
            channels = input_shape[1]
        else:
            channels = input_shape[-1]

        if self.rank == 2:
            posterior_affine_tensor = tfp.distributions.Normal(
                loc=tf.zeros_like(mu_kernel),
                scale=tf.nn.softplus(rho_kernel)).sample()
        if self.rank == 1:
            posterior_affine_tensor = tfp.distributions.Normal(
                loc=tf.zeros_like(mu_kernel),
                scale=tf.nn.softplus(rho_kernel)).sample()

        if self.rank == 2:
            outputs = self._convolution_op(
                input, mu_kernel)
        if self.rank == 1:
            outputs = self._convolution_op(
                input, mu_kernel)

        sign_input = random_rademacher(
            tf.concat([batch_shape,
                       tf.expand_dims(channels, 0)], 0),
            dtype=input.dtype)
        sign_output = random_rademacher(
            tf.concat([batch_shape,
                       tf.expand_dims(self.n_filters, 0)], 0),
            dtype=input.dtype)

        if self.data_format == 'channels_first':
            for _ in range(self.rank):
                sign_input = tf.expand_dims(sign_input, -1)  # 2D ex: (B, C, 1, 1)
                sign_output = tf.expand_dims(sign_output, -1)
        else:
            for _ in range(self.rank):
                sign_input = tf.expand_dims(sign_input, 1)  # 2D ex: (B, 1, 1, C)
                sign_output = tf.expand_dims(sign_output, 1)

        perturbed_inputs = self._convolution_op(
            input * sign_input, posterior_affine_tensor) * sign_output

        outputs += perturbed_inputs
        if self.use_bias:
            outputs = nn.bias_add(outputs, bias, data_format=self.tf_data_format)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = {
            'rank': self.rank,
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer)
        }
        base_config = super(_VariationalConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class VariationalConv1D(_VariationalConv):
    """1D convolution layer (e.g. temporal convolution).

    # Input shape
        3D tensor with shape: `(batch_size, steps, input_dim)`

    # Output shape
        3D tensor with shape: `(batch_size, new_steps, n_filters)`
        `steps` value might have changed due to padding or strides.
    """

    def __init__(self,
                 n_filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        if padding == 'causal':
            if data_format != 'channels_last':
                raise ValueError(
                    'When using causal padding in `VaritionalConv1D`, `data_format` must be "channels_last" (temporal data).')
        super(VariationalConv1D, self).__init__(
            rank=1,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def get_config(self):
        config = super(VariationalConv1D, self).get_config()
        config.pop('rank')
        return config


class VariationalConv2D(_VariationalConv):
    """2D convolution layer (e.g. spatial convolution over images).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, rows, cols, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        4D tensor with shape:
        `(samples, n_filters, new_rows, new_cols)`
        if `data_format` is `"channels_first"`
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, n_filters)`
        if `data_format` is `"channels_last"`.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 n_filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_initializer='he_normal',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(VariationalConv2D, self).__init__(
            rank=2,
            n_filters=n_filters,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=4)

    def get_config(self):
        config = super(VariationalConv2D, self).get_config()
        config.pop('rank')
        return config


class VariationalConv3D(_VariationalConv):
    """3D convolution layer (e.g. spatial convolution over volumes).

    # Input shape
        5D tensor with shape:
        `(samples, channels, conv_dim1, conv_dim2, conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, conv_dim1, conv_dim2, conv_dim3, channels)`
        if `data_format` is `"channels_last"`.

    # Output shape
        5D tensor with shape:
        `(samples, n_filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`
        if `data_format` is `"channels_first"`
        or 5D tensor with shape:
        `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, n_filters)`
        if `data_format` is `"channels_last"`.
        `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have changed due to padding.
    """

    def __init__(self,
                 n_filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super(VariationalConv3D, self).__init__(
            rank=3,
            n_filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            activity_regularizer=activity_regularizer,
            **kwargs)
        self.input_spec = InputSpec(ndim=5)

    def get_config(self):
        config = super(VariationalConv3D, self).get_config()
        config.pop('rank')
        return config
