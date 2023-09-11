#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer
import tensorflow_probability as tfp
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import activations
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.eager import context
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import gen_math_ops

tfd = tfp.distributions
kl = tfd.kullback_leibler


class VariationalLayer(Layer):

    def __init__(self, units, kernel_regularizer=None, kernel_constraint=None,
                 use_bias=True, activity_regularizer=None, kernel_initializer="glorot_uniform",
                 activation=None, seed=None, **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(VariationalLayer, self).__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if not isinstance(units, int) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.units = units
        self.seed = seed
        self.kernel_initializer = tf.keras.initializers.HeNormal(seed=self.seed)
        self.kernel_regularizer = tf.keras.regularizers.L1L2(l1=1e-6, l2=1e-6)
        self.kernel_constraint = kernel_constraint
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `Dense` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError('The last dimension of the inputs to `StickbreakingLayer` '
                             'should be defined. Found `None`.')

        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        # Instantiate/initialize the variational priors

        # Posterior Weights
        if self.use_bias:
            self.bias = self.add_weight(shape=self.units,
                                        initializer='zeros',
                                        name='post_bias',
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)

        log_var_init = tf.keras.initializers.RandomNormal(mean=-6.0, stddev=0.01, seed=self.seed)
        self.q_mu_log_var = self.add_weight(shape=[self.units, ],
                                            initializer=log_var_init,
                                            name='post_q_mu_log_var',
                                            regularizer=self.kernel_regularizer,
                                            constraint=self.kernel_constraint)

        self.components_q_mu_kernels = self.add_weight(shape=[last_dim, self.units],
                                                       initializer=self.kernel_initializer,
                                                       name='post_components_q_mu_kernels',
                                                       regularizer=self.kernel_regularizer,
                                                       constraint=self.kernel_constraint)

        if self.use_bias:
            self.prior_bias = self.add_weight(shape=self.units,
                                              initializer='zeros',
                                              name='prior_bias',
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint,
                                              trainable=False)

        # q_mu_log_var_init = tf.keras.initializers.RandomNormal(mean=-6.0, stddev=0.01)
        self.prior_q_mu_log_var = self.add_weight(shape=[self.units, ],
                                                  initializer=log_var_init,
                                                  name='prior_q_mu_log_var',
                                                  regularizer=self.kernel_regularizer,
                                                  constraint=self.kernel_constraint, trainable=False)


        self.prior_components_q_mu_kernels = self.add_weight(shape=[last_dim, self.units],
                                                             initializer=self.kernel_initializer,
                                                             name='prior_components_q_mu_kernels',
                                                             regularizer=self.kernel_regularizer,
                                                             constraint=self.kernel_constraint,
                                                             trainable=False)

    def call(self, inputs, *args, **kwargs):

        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

        # Specify the variational distributions q(theta), q(mu), and q(tau)
        self.p_w_d = tfd.MultivariateNormalDiag(
            loc=self.components_q_mu_kernels,
            scale_diag=tf.nn.softplus(self.q_mu_log_var) + 1e-6,
            validate_args=True
        )

        out_w = self.p_w_d.sample()
        out, log_prob = self.apply_kernel(inputs, out_w, self.p_w_d)
        if self.activation is not None:
            out = self.activation(out)

        # DKL losses
        self.prior_p_w_d = tfd.MultivariateNormalDiag(
            loc=self.prior_components_q_mu_kernels,
            scale_diag=tf.nn.softplus(self.prior_q_mu_log_var) + 1e-6,
            validate_args=True
        )
        #prior_out_w = self.prior_p_w_d.sample()
        #prior_out, prior_log_prob = self.apply_kernel(inputs, prior_out_w, self.prior_p_w_d)

        w_dkl = tfd.kl_divergence(self.p_w_d, self.prior_p_w_d)
        self.add_loss(losses=tf.reduce_sum(w_dkl))
        return out#, prior_out

    def apply_kernel(self, input_tensor, kernel, p_w_d):
        out = tf.matmul(input_tensor, kernel)  # , axes=[0,1])
        if self.use_bias:
            out = out + self.bias
        expert_output = out
        log_probs = p_w_d.log_prob(kernel)
        return expert_output, log_probs
