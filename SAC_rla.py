import tensorflow_addons as tfa
from typing import Sequence
from numbers import Number
from SAC_utils import *
from SAC_gen_conf import *
from tensorflow.python.keras.utils import losses_utils

class UMIM:
    def __init__(self, largest_obs_dim, largest_act_dim, discount, polyak_coef, lr,
                 n_hidden_units, save_dir, task, total_episodes):

        self.largest_obs_dim = largest_obs_dim
        self.largest_act_dim = largest_act_dim
        self.n_hidden_units = n_hidden_units
        self.discount = discount
        self.polyak_coef = polyak_coef
        self.lr = lr
        self.save_dir = save_dir
        self.gamma = discount
        self.reward_scale = 1.0
        self.temperature = 0.2
        self.task = task
        self.total_episodes = total_episodes

    def reset_state(self):
        if hasattr(self, 'actor_network'):
            del self.actor_network
            del self.actor_optimizer
            del self.Qs
            del self.Q_targets
            del self.Q_optimizers
            del self.alpha_optimizer

        # alpha optimizer
        self.alpha_lr = self.lr
        self.target_entropy = -np.prod(self.largest_act_dim)
        self.log_alpha = tf.Variable(0.0)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)

        self.alpha_optimizer = tf.keras.optimizers.Adam(self.alpha_lr, name='alpha_optimizer')
        ### Creating networks and optimizers ###
        # Policy network
        # action_output are the squashed actions and action_original those straight from the normal distribution
        logprob_epsilon = 1e-6  # For numerical stability when computing tf.log
        self.actor_network = ActorNetwork(self.n_hidden_units, self.largest_act_dim,
                                          logprob_epsilon,
                                          inputdim=self.largest_obs_dim,
                                          name="%s"%self.task)

        # 2 Soft q-functions networks + targets
        self.softq_network = SoftQNetwork(self.n_hidden_units, qi="source_0",
                                          inputdim=self.largest_obs_dim + self.largest_act_dim,
                                          name="%s"%self.task)
        self.softq_target_network = SoftQNetwork(self.n_hidden_units, qi="target_0",
                                                 inputdim=self.largest_obs_dim + self.largest_act_dim,
                                                 name="%s"%self.task)

        self.softq_network2 = SoftQNetwork(self.n_hidden_units, qi="source_1",
                                           inputdim=self.largest_obs_dim + self.largest_act_dim,
                                           name="%s"%self.task)

        self.softq_target_network2 = SoftQNetwork(self.n_hidden_units, qi="target_1",
                                                  inputdim=self.largest_obs_dim + self.largest_act_dim,
                                                  name="%s"%self.task)

        # Optimizers for the networks
        self.softq_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.softq_optimizer2 = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        self.Qs = [self.softq_network, self.softq_network2]
        self.Q_targets = [self.softq_target_network, self.softq_target_network2]
        self.Q_optimizers = [self.softq_optimizer, self.softq_optimizer2]
        self._update_target(tau=0.0)

        self.updates_performed = 0

    def softq_value(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network(states, actions)

    def softq_value2(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network2(states, actions)

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        if len(np.shape(states)) == 1:
            states = states[None, :]
        return self.actor_network(states)[0]

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return self.actor_network(state[None, :])[0][0]

    def step(self, obs):
        return self.actor_network(obs)[0]

    def get_models(self):
        return [self.actor_network, self.softq_network, self.softq_network2, self.softq_target_network,
                self.softq_target_network2]

    @tf.function(experimental_relax_shapes=True)
    def _update_alpha(self, observations):
        if not isinstance(self.target_entropy, Number):
            return 0.0

        actions, log_pis, _, _ = self.actor_network(observations)
        with tf.GradientTape() as tape:
            alpha_losses = -1.0 * (
                    self.alpha * tf.stop_gradient(log_pis + self.target_entropy))
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)

        alpha_gradients = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(
            alpha_gradients, [self.log_alpha]))

        return alpha_losses

    @tf.function(experimental_relax_shapes=True)
    def td_targets(self, rewards, discounts, next_values):
        return rewards + discounts * next_values

    def get_idx_name(self, name, postflag):
        post = ""
        if postflag:
            post = "post_"
        idx = -1
        # kernels
        if post + "bias" in name:
            idx = 12
        elif post + "expert_embedding_bias" in name:
            idx = 13
        elif post + "2_expert_embedding_bias" in name:
            idx = 14
        elif post + "q_mu_log_var" in name:
            idx = 15
        elif post + "q_tau_rate" in name:
            idx = 16
        elif post + "components_q_mu_kernels" in name:
            idx = 17
        elif post + "components_q_tau_kernels" in name:
            idx = 18
        elif "gamma" in name:
            idx = 19
        elif "beta" in name:
            idx = 20
        elif "moving_mean" in name:
            idx = 21
        elif "moving_variance" in name:
            idx = 22

        return idx

    # @tf.function(experimental_relax_shapes=True)
    def copy_weights(self, old_models, num_train_samples=1, verbose=False):
        new_models = [self.actor_network, self.softq_network, self.softq_network2,
                                                     self.softq_target_network, self.softq_target_network2]
        assert len(old_models) == len(new_models)
        for old_model, new_model in zip(old_models, new_models):
            if old_model is not None:
                if verbose:
                    print("copy weights %s to %s" % (old_model.model_name, new_model.model_name))
                for new_layer, old_layer in zip(new_model.model_layers.layers, old_model.model_layers.layers):
                    old_weights = old_layer.get_weights()
                    new_weights = new_layer.get_weights()
                    old_weight_obj = old_layer.weights
                    new_weight_obj = new_layer.weights
                    prior_weights = [[] for _ in range(100)]
                    prior_weight_names = [[] for _ in range(100)]
                    for wobj, weight in zip(old_weight_obj, old_weights):
                        idx = self.get_idx_name(wobj.name, postflag=True)
                        if idx >= 0:
                            prior_weights[idx] = weight
                            prior_weight_names[idx] = wobj.name
                            if verbose:
                                print("old model: detected", prior_weight_names[idx], "and saved on slot", idx)
                        elif "prior" not in wobj.name:
                            print("old model: -> did not recognize", wobj.name)
                        elif "post" in wobj.name:
                            print("\033[91m WARNING: could not find entry of posterior weight %s  \033[0m" % wobj.name)
                    updated_new_weights = []

                    for wobj, weight, old_wobj, old_weight in zip(new_weight_obj, new_weights, old_weight_obj,
                                                                  old_weights):
                        idx = self.get_idx_name(wobj.name, postflag=False)

                        if idx >= 0:
                            updated_new_weights.append(prior_weights[idx])
                            if verbose:
                                print("new model: detected", wobj.name, "and overwriting using",
                                      prior_weight_names[idx])
                        else:
                            updated_new_weights.append(old_weight)
                            if "prior" in wobj.name:
                                print("\033[91m WARNING: could not find weights for prior %s  \033[0m" % wobj.name)
                            if verbose:
                                print("new model: could not find %s and copied weights of %s" % (
                                wobj.name, old_wobj.name))
                    if len(updated_new_weights):
                        new_layer.set_weights(updated_new_weights)

    @tf.function(experimental_relax_shapes=True)
    def _update_target(self, tau):
        for Q, Q_target in zip(self.Qs, self.Q_targets):
            for source_weight, target_weight in zip(
                    Q.trainable_variables, Q_target.trainable_variables):
                target_weight.assign(
                    tau * target_weight + (1.0 - tau) * source_weight)

    @tf.function(experimental_relax_shapes=True)
    def compute_Q_targets(self, next_Q_values,
                          next_log_pis,
                          rewards,
                          terminals,
                          discount,
                          entropy_scale,
                          reward_scale):
        next_values = next_Q_values - entropy_scale * next_log_pis
        terminals = tf.cast(terminals, next_values.dtype)

        Q_targets = self.td_targets(
            rewards=reward_scale * rewards,
            discounts=discount,
            next_values=(1.0 - terminals) * next_values)
        return Q_targets

    @tf.function(experimental_relax_shapes=True)
    def _compute_Q_targets(self, batch):
        next_observations = batch['states1']
        rewards = batch['rewards']
        terminals = batch['terminals1']

        entropy_scale = tf.convert_to_tensor(self.alpha)
        reward_scale = tf.convert_to_tensor(self.reward_scale)
        discount = tf.convert_to_tensor(self.gamma)

        next_actions, next_log_pis, _, _ = self.actor_network(next_observations)
        next_Qs_values = []
        for Q in self.Q_targets:
            next_Qs_values.append(Q(next_observations, next_actions))
        next_Qs_values = tf.concat(next_Qs_values, axis=-1)
        next_Qs_values = tf.math.reduce_min(next_Qs_values, axis=-1)

        Q_targets = self.compute_Q_targets(
            next_Qs_values,
            next_log_pis,
            rewards,
            terminals,
            discount,
            entropy_scale,
            reward_scale)
        tf.debugging.assert_all_finite(Q_targets, "q targets not finite")
        return tf.stop_gradient(Q_targets)

    def measure_graph_size(self, f, *args):
        g = f.get_concrete_function(*args).graph
        print("{}({}) contains {} nodes in its graph".format(
            f.__name__, ', '.join(map(str, args)), len(g.as_graph_def().node)))


    @tf.function
    def update_model(self, sample, aug_samples, batch_size):
        observations = sample["states0"]
        aug_observations = aug_samples["states0"]
        actions = sample["actions"]

        # Computing target for q-functions
        softq_targets = self._compute_Q_targets(sample)
        softq_targets = tf.reshape(softq_targets, [batch_size, 1])
        tf.debugging.assert_all_finite(softq_targets, "q values not finite")
        q_losses = []
        avg_td_errors = tf.zeros_like(softq_targets)
        avg_old_td_errors = tf.zeros_like(softq_targets)
        avg_qmodel_losses = tf.zeros(shape=())
        for Q, optimizer in zip(self.Qs, self.Q_optimizers):
            with tf.GradientTape() as tape:
                Q_values, q_model_losses = Q.loss_call(observations, actions)
                avg_qmodel_losses = avg_qmodel_losses + tf.multiply(0.5, q_model_losses)
                pred_Q_losses = tf.keras.losses.huber(y_true=softq_targets, y_pred=Q_values)
                Q_losses = tf.nn.compute_average_loss(pred_Q_losses)
                avg_td_errors = avg_td_errors + 0.5 * tf.abs(softq_targets - Q_values)
                q_losses.append(Q_losses)
            total_gradients = tape.gradient(Q_losses, Q.trainable_variables)
            [tf.debugging.assert_all_finite(g, "q fun grads not finite") for g in total_gradients]
            optimizer.apply_gradients(zip(total_gradients, Q.trainable_variables))
        entropy_scale = tf.convert_to_tensor(self.alpha)
        with tf.GradientTape() as actor_tape:
            actions, log_pis, mean, std, actor_model_loss, contr = self.actor_network.loss_call(observations)
            _, _, _, _, _, aug_contr = self.actor_network.loss_call(aug_observations)
            Qs_log_targets = []
            for Q in self.Qs:
                Qs_log_targets.append(Q(observations, actions))
            Qs_log_targets = tf.concat(Qs_log_targets, axis=-1)
            Qs_log_targets = tf.math.reduce_min(Qs_log_targets, axis=-1)
            contr_loss = self.actor_network.add_contrastive_loss(contr, aug_contr)
            actor_loss = (entropy_scale * log_pis) - Qs_log_targets + contr_loss
            actor_loss = tf.nn.compute_average_loss(actor_loss)
        total_actor_gradients = actor_tape.gradient(actor_loss, self.actor_network.trainable_weights)
        [tf.debugging.assert_all_finite(g, "actor grads not finite") for g in total_actor_gradients]
        self.actor_optimizer.apply_gradients(zip(total_actor_gradients, self.actor_network.trainable_weights))

        self._update_target(tau=self.polyak_coef)

        self._update_alpha(observations=observations)

        return tf.reduce_mean(Q_losses), tf.reduce_mean(actor_loss), tf.reduce_mean(log_pis), tf.reduce_mean(
            softq_targets), tf.reduce_mean(Q_values), tf.reduce_mean(Qs_log_targets), tf.reduce_mean(actor_model_loss), \
               tf.reduce_mean(avg_qmodel_losses), self.alpha, mean, std, avg_td_errors, avg_old_td_errors

