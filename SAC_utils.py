import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, InputLayer, BatchNormalization, Input, Concatenate, LayerNormalization
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
from tensorflow.python.keras.utils import losses_utils
from collections import deque
from BayesianDenseMoe import *
from VariationalLayer import VariationalLayer


class ReplayBuffer(object):
    def __init__(self, buffer_size: int) -> None:
        self.buffer_size: int = buffer_size
        self.num_experiences: int = 0
        self.buffer: deque = deque()

    def get_batch(self, num_samples: int) -> np.array:
        # Randomly sample batch_size examples
        experiences = random.sample(self.buffer, num_samples)
        return {
            "states0": np.asarray([exp[0] for exp in experiences], np.float32),
            "actions": np.asarray([exp[1] for exp in experiences], np.float32),
            "rewards": np.asarray([exp[2] for exp in experiences], np.float32),
            "states1": np.asarray([exp[3] for exp in experiences], np.float32),
            "terminals1": np.asarray([exp[4] for exp in experiences], np.float32)
        }

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        # print("Added Experience:", experience, "Count is now %d"%self.num_experiences)

    @property
    def size(self) -> int:
        return self.buffer_size

    @property
    def n_entries(self) -> int:
        # If buffer is full, return buffer size
        # Otherwise, return experience counter
        return self.num_experiences

    def get_last_n(self, n):
        if n > self.num_experiences:
            n = self.num_experiences
        data = []
        for _ in range(n):
            data.append(self.buffer.popleft())
        return data

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def extend(self, data):
        self.num_experiences += len(data)
        self.buffer.extend(data)


class ActorNetwork(Model):
    def __init__(self, n_hidden_units, n_actions, logprob_epsilon, inputdim,
                 n_experts=1, vmoe=True, k=1, name="0_0"):
        super(ActorNetwork, self).__init__()
        self.n_hidden_units = n_hidden_units
        self.deep = True
        self.vmoe = vmoe
        self.n_actions = n_actions
        self.n_experts = n_experts
        self.temperature = 0.2
        self.n_proj_units = 128
        self.gating_entropy_beta = 1e-1
        self.act_fun = tf.nn.leaky_relu
        self.proj_fun = tf.nn.tanh
        self.logprob_epsilon = logprob_epsilon
        self.k = k
        self.model_name = "actor_"+name
        input_layer = Input(shape=inputdim)
        x = VariationalLayer(units=self.n_hidden_units,
                             activation=self.act_fun,
                             name='input_%s' % self.model_name)(input_layer)
        x = LayerNormalization()(x)
        # x = Concatenate(axis=-1)([x, input_layer])
        x = VariationalLayer(units=self.n_hidden_units,
                             activation=self.act_fun,
                             name='hidden_0_%s' % self.model_name)(x)
        #x = Concatenate(axis=-1)([x, input_layer])
        x = LayerNormalization()(x)
        actor_projection = VariationalLayer(units=self.n_proj_units,
                                            activation=self.proj_fun,
                                            name='projection_layer_%s' % self.model_name)(x)
        actor_output = VariationalLayer(units=self.n_actions * 2,
                                        activation=None,
                                        name='output_%s' % self.model_name)(x)
        self.model_layers = Model(input_layer, [actor_output, actor_projection])
        self.model_layers.compile()
        print(self.model_layers.summary())

    @tf.function(experimental_relax_shapes=True)
    def gaussian_likelihood(self, input, mu, log_std):
        """
        Helper to compute log likelihood of a gaussian.
        Here we assume this is a Diagonal Gaussian.
        :param input_: (tf.Tensor)
        :param mu_: (tf.Tensor)
        :param log_std: (tf.Tensor)
        :return: (tf.Tensor)
        """
        pre_sum = -0.5 * (
                ((input - mu) / (tf.exp(log_std) + self.logprob_epsilon)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    @tf.function(experimental_relax_shapes=True)
    def call(self, inp):
        x, contr = self.model_layers(inp)
        mean = x[:, :self.n_actions]
        log_std = x[:, self.n_actions:]
        log_std = tf.clip_by_value(t=log_std, clip_value_min=-20, clip_value_max=2)
        std = tf.exp(log_std)
        action = mean + tf.random.normal(tf.shape(mean)) * std
        squashed_actions = tf.tanh(action)
        # numerically unstable:
        # logprob = action_dist.log_prob(action) - tf.reduce_sum(
        #     tf.math.log((1.0 - tf.pow(squashed_actions, 2)) + self.logprob_epsilon), axis=-1)
        # ref: https://github.com/vitchyr/rlkit/blob/0073d73235d7b4265cd9abe1683b30786d863ffe/rlkit/torch/distributions.py#L358
        # ref: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L73
        # numerically stable version:
        # logprob = self.gaussian_likelihood(action, mean, log_std) - tf.reduce_sum((2 * (np.log(2) - squashed_actions - tf.math.softplus(-2 * squashed_actions))), axis=-1)
        logprob = self.gaussian_likelihood(mu=mean, input=action, log_std=log_std) - tf.reduce_sum(
            tf.math.log((1.0 - tf.pow(squashed_actions, 2)) + self.logprob_epsilon), axis=-1)
        return squashed_actions, logprob, tf.nn.tanh(mean), std

    @tf.function(experimental_relax_shapes=True)
    def loss_call(self, inp):
        x, contr = self.model_layers(inp)
        mean = x[:, :self.n_actions]
        log_std = x[:, self.n_actions:]
        log_std = tf.clip_by_value(t=log_std, clip_value_min=-20, clip_value_max=2)
        std = tf.exp(log_std)
        action = mean + tf.random.normal(tf.shape(mean)) * std
        squashed_actions = tf.tanh(action)
        logprob = self.gaussian_likelihood(mu=mean, input=action, log_std=log_std) - tf.reduce_sum(
            tf.math.log((1.0 - tf.pow(squashed_actions, 2)) + self.logprob_epsilon), axis=-1)
        return squashed_actions, logprob, tf.nn.tanh(mean), std, tf.reduce_sum(self.model_layers.losses), contr

    @tf.function(experimental_relax_shapes=True)
    def total_similarity(self, hidden_anchor, hidden_pos, hidden_neg):
        contrastive_loss = self.add_contrastive_loss(hidden1=hidden_anchor, hidden2=hidden_pos,
                                                     hidden_norm=True)
        contrastive_loss_neg = self.add_contrastive_loss(hidden1=hidden_anchor, hidden2=hidden_neg,
                                                         hidden_norm=True)
        contrastive_loss = contrastive_loss[:, tf.newaxis]
        contrastive_loss_neg = contrastive_loss_neg[:, tf.newaxis]
        return -tf.reduce_mean(tf.concat([contrastive_loss, contrastive_loss_neg], axis=-1),
                               axis=-1, keepdims=True)

    @tf.function(experimental_relax_shapes=True)
    def add_contrastive_loss(self, hidden1, hidden2,
                             hidden_norm=True):
        """Compute loss for model.
        https://github.com/google-research/simclr/blob/master/tf2/objective.py
        Args:
          hidden: hidden vector (`Tensor`) of shape (bsz, dim).
          hidden_norm: whether to use normalization on the hidden vector.
          temperature: a `floating` number for temperature scaling.
          strategy: context information for tpu.
        Returns:
          A loss scalar.
          The logits for contrastive prediction task.
          The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        LARGE_NUM = 1e9
        if hidden_norm:
            hidden1 = tf.math.l2_normalize(hidden1, -1)
            hidden2 = tf.math.l2_normalize(hidden2, -1)
        batch_size = tf.shape(hidden1)[0]

        labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
        masks = tf.one_hot(tf.range(batch_size), batch_size)

        logits_aa = tf.matmul(hidden1, hidden1, transpose_b=True) / self.temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = tf.matmul(hidden2, hidden2, transpose_b=True) / self.temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = tf.matmul(hidden1, hidden2, transpose_b=True) / self.temperature
        logits_ba = tf.matmul(hidden2, hidden1, transpose_b=True) / self.temperature
        sm_xent = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                          reduction=losses_utils.ReductionV2.NONE)
        pred_a = tf.concat([logits_ab, logits_aa], 1)
        pred_b = tf.concat([logits_ba, logits_bb], 1)
        loss_a = sm_xent(labels, pred_a)
        loss_b = sm_xent(labels, pred_b)
        loss = loss_a + loss_b
        return loss


    @tf.function
    def get_model_losses(self):
        return tf.reduce_sum(self.model_layers.losses)

    def _get_params(self):
        with self.graph.as_default():
            params = tf.trainable_variables()
        names = [p.name for p in params]
        values = self.sess.run(params)
        params = {k: v for k, v in zip(names, values)}
        return params

    def __getstate__(self):
        params = self._get_params()
        state = self.args_copy, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        self.restore_params(params)


class SoftQNetwork(Model):
    def __init__(self, n_hidden_units, inputdim, qi="0", n_experts=1,
                 vmoe=True, k=1, name="0_0"):
        super(SoftQNetwork, self).__init__()
        self.deep = True
        self.vmoe = vmoe
        self.n_hidden_units = n_hidden_units
        self.k = k
        self.act_fun = tf.nn.leaky_relu
        self.gating_entropy_beta = 1e-1
        self.model_name = "qnet_"+name+"_%s" % qi
        self.kl_divergence_function = (lambda q, p: ds.kl_divergence(q, p) / tf.cast(1.0, dtype=tf.float32))
        self.entropy_function = (lambda p: tf.maximum(p.entropy(), 0.0) / tf.cast(10000.0, dtype=tf.float32))
        input_layer = Input(shape=inputdim)
        x = VariationalLayer(units=self.n_hidden_units,
                             activation=self.act_fun,
                             name='input_%s' % self.model_name)(input_layer)
        # x = Concatenate(axis=-1)([x, input_layer])
        x = LayerNormalization()(x)
        x = VariationalLayer(units=self.n_hidden_units,
                             activation=self.act_fun,
                             name='hidden_0_%s' % self.model_name)(x)
        x = LayerNormalization()(x)
        # x = Concatenate(axis=-1)([x, input_layer])
        # x = VariationalLayer(units=self.n_hidden_units,
        #                      activation=self.act_fun,
        #                      name='hidden_1_%s' % self.model_name)(x)
        # x = Concatenate(axis=-1)([x, input_layer])
        # x = VariationalLayer(units=self.n_hidden_units,
        #                      activation=self.act_fun,
        #                      name='hidden_2_%s' % self.model_name)(x)
        # x = Concatenate(axis=-1)([x, input_layer])
        out = VariationalLayer(units=1,
                               activation=None,
                               name='output_%s' % self.model_name)(x)
        self.model_layers = Model(input_layer, out)
        self.model_layers.compile()
        print(self.model_layers.summary())

    @tf.function
    def get_model_losses(self):
        return tf.reduce_sum(self.model_layers.losses)

    @tf.function(experimental_relax_shapes=True)
    def call(self, states, actions):
        x = tf.concat([states, actions], -1)
        return self.model_layers(x)

    @tf.function(experimental_relax_shapes=True)
    def loss_call(self, states, actions):
        x = tf.concat([states, actions], -1)
        return self.model_layers(x), tf.reduce_sum(self.model_layers.losses)

def task_mi_from_mat(mat):
    q = np.mean(mat, axis=0) + 1e-3
    task_mi = 0.0
    for p in mat:
        task_mi = task_mi + np.sum(p * np.log(p / q))
    return (task_mi / mat.shape[0]) / np.log(2.0)

def plot_episode_stats(actor_losses, softq_losses, action_logprob_means, episode_rewards, smoothing_window=15):
    # Plot the episode length over time
    plt.figure(figsize=(10, 5))
    plt.plot(actor_losses, label="Actor")
    plt.xlabel("Episode")
    plt.ylabel("Actor Loss")
    plt.title("Actor Loss over Time")

    plt.figure(figsize=(10, 5))
    plt.plot(softq_losses, label="Soft-Q")
    plt.xlabel("Episode")
    plt.ylabel("Soft-Q Loss")
    plt.title("Soft-Q Loss over Time")

    plt.figure(figsize=(10, 5))
    plt.plot(action_logprob_means, label="log(p(a))")
    plt.xlabel("Episode")
    plt.ylabel("Log Prob")
    plt.title("Log Prob over Time")

    # Plot the episode reward over time
    plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show()


def plot_reward(episode_rewards):
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time")
    plt.show()