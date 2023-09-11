from SAC_utils import *
from timeit import default_timer as timer
import time
from parallel_gym import make_mp_envs
from tensorflow_probability import distributions as tfd
import tensorflow.keras.backend as K

class Agent:
    def __init__(self, models, replay_buffers, replay_start_size,
                 batch_size, train_n_steps, largest_act_dim, largest_obs_dim, task, train_interval):
        self.models = models
        self.replay_buffers = replay_buffers
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.largest_act_dim = largest_act_dim
        self.largest_obs_dim = largest_obs_dim
        self.batch_size = batch_size
        self.train_n_steps = train_n_steps
        self.n_timesteps = 1000
        self.train_interval = train_interval
        self.eval_interval = 10000
        self.total_steps = 0
        self.total_episodes = 0
        self.temperature = 0.2
        self.aug_noise_std = 1e-3
        self.task = task

    def reset_state(self, train_env, n_actions, eval_envs):
        if hasattr(self, "train_env"):
            del self.train_env
            del self.eval_envs
        self.train_env = train_env
        self.n_actions = n_actions
        self.eval_envs = eval_envs

    def augment_samples(self, samples):
        noise = tf.random.normal(mean=tf.zeros_like(samples["states0"]),
                                 stddev=tf.zeros_like(samples["states0"]) + self.aug_noise_std,
                                 shape=tf.shape(samples["states0"]))
        samples["states0"] = samples["states0"] + noise
        return samples

    def augment_sample(self, sample):
        noise = tf.random.normal(mean=tf.zeros_like(sample), stddev=tf.zeros_like(sample) + self.aug_noise_std,
                                 shape=tf.shape(sample))
        sample = sample + noise
        return sample

    def train(self):
        train_step = 0

        # Noise + epsilon parameters
        epsilon = 1
        train_interval = self.train_interval
        eval_interval = self.eval_interval
        num_env = 128
        print("Collecting with %d envs" % num_env)
        sample_eval_ctr = 0
        actor_losses = []
        softq_losses = []
        action_logprob_means = []
        env_evals_r = [[] for _ in range(len(self.eval_envs))]
        durations = deque(maxlen=20)
        parallel_envs = make_mp_envs(env_id=self.train_env.envs[0].spec.id, num_env=num_env, norm_reward=True,
                                     largest_obs_dim=self.largest_obs_dim)
        train_sample_ctr = 0
        duration_sample_ctr = 0
        duration = .0
        action_noise_std = 0.1
        while train_step < self.train_n_steps:
            start = timer()
            state = parallel_envs.reset().astype(np.float32)
            state = tf.constant(state)
            for k in range(self.n_timesteps):
                xxp = self.augment_sample(state)
                xxn = self.augment_sample(state)
                cool_down = train_step - 0.2 * self.train_n_steps
                dispatched_states, _, _, _, _, _, dispatcher = self.dispatch(x=state,
                                                                             xxp=xxp,
                                                                             xxn=xxn,
                                                                             first_run=self.task == 0,
                                                                             new_model_cooldown=cool_down)
                dispatched_actions = [[] for _ in range(len(dispatched_states))]
                all_actions = []
                for mi in range(len(dispatched_states)):
                    if len(dispatched_states[mi]):
                        action = self.models[mi].actions(dispatched_states[mi])
                        action = action + np.random.normal(loc=0.0, scale=action_noise_std, size=np.shape(action))
                        ext_action = action[:, :self.n_actions]
                        ext_action = np.reshape(ext_action,
                                                newshape=(np.shape(ext_action)[0], 1, np.shape(ext_action)[-1]))
                        dispatched_actions[mi].extend(action)
                        all_actions.append(ext_action)

                all_actions = np.hstack(all_actions)
                new_state, reward, done, _ = parallel_envs.step(all_actions)

                if dispatcher is not None:
                    dispatched_rewards = dispatcher.dispatch(reward)
                    dispatched_new_states = dispatcher.dispatch(new_state)
                    dispatched_dones = dispatcher.dispatch(done)
                else:
                    dispatched_rewards = [reward]
                    dispatched_new_states = [new_state]
                    dispatched_dones = [done]

                for states, actions, rewards, next_states, dones, mi in zip(dispatched_states, dispatched_actions,
                                                                            dispatched_rewards, dispatched_new_states,
                                                                            dispatched_dones, range(len(self.models))):
                    for s, a, r, ss, d in zip(states, actions, rewards, next_states, dones):
                        self.replay_buffers[mi].add(state=s, action=a, reward=r, new_state=ss, done=d)

                state = new_state
                train_step += np.shape(state)[0]
                train_sample_ctr += np.shape(state)[0]
                sample_eval_ctr += np.shape(state)[0]
                duration_sample_ctr += np.shape(state)[0]
                if train_sample_ctr >= train_interval:
                    self.total_episodes += 1
                    break
            if train_sample_ctr >= train_interval:
                K.set_learning_phase(True)
                mean_a_loss = 0.0
                mean_sq_loss = 0.0
                mean_alp_loss = 0.0
                mean_soft_targets_mean = 0.0
                mean_q_values_mean = 0.0
                mean_Qs_log_targets = 0.0
                mean_alpha = 0.0
                train_sample_ctr = 0
                mean_a_model_loss = 0.0
                mean_sq_model_loss = 0.0
                num_updates = 50  # train_interval #np.minimum(100, episode_length)

                w_idxs = [[] for _ in range(len(self.models))]
                new_prios = [[] for _ in range(len(self.models))]
                for nu in range(num_updates):
                    for mi in range(len(self.models)):
                        if self.replay_buffers[mi].n_entries < self.replay_start_size:
                            continue
                        sample, w, w_idx = self.replay_buffers[mi].get_batch(global_step=train_step,
                                                                             num_samples=self.batch_size)
                        aug_samples = self.augment_samples(sample)
                        softq_loss, actor_loss, action_logprob_mean, soft_targets_mean, q_values_mean, \
                        Qs_log_targets, actor_model_losses, qmodel_losses, alpha, means, stds, td_error, old_td_error = \
                        self.models[mi].update_model(
                            sample,
                            aug_samples,
                            self.batch_size)
                        w_idxs[mi].extend(w_idx)
                        td_error = td_error.numpy()[:, 0]
                        new_prios[mi].extend(td_error)
                        mean_a_loss += np.array(actor_loss) / len(self.models)
                        mean_a_model_loss += np.array(actor_model_losses) / len(self.models)
                        mean_sq_model_loss += np.array(qmodel_losses) / len(self.models)
                        mean_sq_loss += np.array(softq_loss) / len(self.models)
                        mean_alp_loss += np.array(action_logprob_mean) / len(self.models)
                        mean_soft_targets_mean += np.array(soft_targets_mean) / len(self.models)
                        mean_q_values_mean += np.array(q_values_mean) / len(self.models)
                        mean_Qs_log_targets += np.array(Qs_log_targets) / len(self.models)
                        mean_alpha += np.array(alpha) / len(self.models)
                        self.models[mi].updates_performed += 1

                for mi in range(len(self.models)):
                    self.replay_buffers[mi].update_priority(indices=w_idxs[mi], priorities=new_prios[mi])
                mean_a_loss /= train_interval
                mean_sq_loss /= train_interval
                mean_alp_loss /= train_interval
                mean_soft_targets_mean /= train_interval
                mean_q_values_mean /= train_interval
                mean_Qs_log_targets /= train_interval
                mean_alpha /= train_interval
                actor_losses.append(mean_a_loss)
                softq_losses.append(mean_sq_loss)
                action_logprob_means.append(mean_alp_loss)

                print("Environment is", self.train_env.envs[0].spec.id)
                print("Actor loss is", mean_a_loss)
                print("Actor model loss is", mean_a_model_loss)
                print("Action log-p mean is", mean_alp_loss)
                print("Q loss is", mean_sq_loss)
                print("Q model loss is", mean_sq_model_loss)
                print("Soft Q Targets mean", mean_soft_targets_mean)
                print("Q Values mean", mean_q_values_mean)
                print("Q Log Targets mean", mean_Qs_log_targets)
                print("Alpha", mean_alpha)
                print("Epsilon", epsilon)
                print("Buffer Memory: ", [m.n_entries for m in self.replay_buffers])
                print("Models: %d" % (len(self.models)))
                if len(durations):
                    mean_duration = np.mean(durations)
                    time_str = time.strftime('%H:%M:%S',
                                             time.gmtime(mean_duration * (self.train_n_steps - (train_step + 1.0))))
                    print("duration", duration)
                    std_duration = np.std(durations)
                    print("T per Step: %.4f +/- %.4f" % (mean_duration, std_duration))
                    time_str_std = time.strftime('%H:%M:%S',
                                                 time.gmtime(std_duration * (self.train_n_steps - (train_step + 1.0))))
                    print("ETA: %s, +/- %s" % (time_str, time_str_std))
            eval_t = 0
            last_eval_rewars = []
            mi = .0
            if sample_eval_ctr >= eval_interval:
                K.set_learning_phase(False)
                sample_eval_ctr = 0
                # eval
                env_actions = [(0, 6), (1, 6), (2, 8), (3, 1), (4, 3), (5, 7), (6, 7), (7, 17), (8, 7), (9, 1), (10, 2)]
                last_eval_rewars = []
                task_to_gate_matrix = np.ones(shape=(1+self.task, len(self.models)))
                for env_idx, env in enumerate(self.eval_envs):
                    eval_n_actions = env_actions[env_idx][1]
                    # for ei in range(eval_runs):
                    env_running_mask = np.ones(shape=env.no_of_envs)
                    state = env.reset().astype(np.float32)
                    eval_t = 0
                    eval_rewards = [[] for _ in range(env.no_of_envs)]
                    while np.any(env_running_mask):
                        xxp = self.augment_sample(state)
                        xxn = self.augment_sample(state)
                        dispatched_states, _, _, _, _, _, dispatcher = self.dispatch(x=state,
                                                                                     xxp=xxp,
                                                                                     xxn=xxn,
                                                                                     eval=True)
                        dispatched_actions = [[] for _ in range(len(dispatched_states))]
                        all_actions = []
                        for mi in range(len(dispatched_states)):
                            if len(dispatched_states[mi]):
                                action = self.models[mi].actions(dispatched_states[mi])
                                ext_action = action[:, :eval_n_actions]
                                ext_action = np.reshape(ext_action,
                                                        newshape=(np.shape(ext_action)[0], 1, np.shape(ext_action)[-1]))
                                dispatched_actions[mi].extend(action)
                                all_actions.append(ext_action)
                                task_to_gate_matrix[env_idx, mi] += len(dispatched_states[mi])
                        new_state, reward, eval_done, _ = env.step(all_actions[0])
                        state = new_state
                        eval_t += 1
                        for di, d in enumerate(env_running_mask):
                            if d:
                                eval_rewards[di].append(reward[di])
                            else:
                                eval_rewards[di].append(0)
                        for di, d in enumerate(eval_done):
                            if d:
                                env_running_mask[di] = 0
                    eval_rewards = np.array(eval_rewards)
                    eval_rewards = np.sum(eval_rewards, axis=-1)
                    mean_eval_episode_reward = np.max(eval_rewards, axis=-1)
                    env_evals_r[env_idx].append(mean_eval_episode_reward)
                    last_eval_rewars.append(mean_eval_episode_reward)
                for t in range(np.shape(task_to_gate_matrix)[0]):
                    task_to_gate_matrix[t, :] = task_to_gate_matrix[t, :] / np.sum(task_to_gate_matrix[t, :])
                print(task_to_gate_matrix)
                mi = task_mi_from_mat(task_to_gate_matrix)
            end = timer()
            duration = end - start
            durations.append(duration / duration_sample_ctr)
            duration_sample_ctr = 0
            print("env:", self.train_env.envs[0].spec.id
                  , "Episode n.", self.total_episodes, "ended! Steps:", train_step, "The eval rewards are",
                  last_eval_rewars,
                  ", number of steps:", eval_t, ", MI: %.2f" %mi, end="\r")
        parallel_envs.close()
        del parallel_envs
        return env_evals_r

    def test(self, model_path):
        self.model.load(model_path)
        while True:
            obs, done = self.test_env.reset(), False
            while not done:
                action = self.model.action(obs.astype(np.float32))
                obs, reward, done, info = self.test_env.step(action)
                self.test_env.render()

    # @tf.function(experimental_relax_shapes=True)
    def dispatch(self, x, xxp=None, xxn=None, eval=False, first_run=False, max_num_models=5,
                 similarity_threshold=0.75, new_model_cooldown=0, task_id=0):
        model_loss_per_input = []
        large_num = 1e9
        all_samples = tf.concat([x, xxp, xxn], axis=0)
        const_class_accs = [0.0, 0.0, 0.0]
        if len(self.models) > 1 or not first_run or eval:
            if not eval and not first_run and new_model_cooldown <= 0:
                for mi, model in enumerate(self.models):
                    if mi == len(self.models)-1:
                        values = tf.ones((tf.shape(x)[0], 1))
                    else:
                        values = tf.zeros((tf.shape(x)[0], 1))
                    model_loss_per_input.append(values)
            else:
                for mi, model in enumerate(self.models):
                    _, _, _, _, _, embeddings = model.actor_network.loss_call(all_samples)
                    proj_latent_anch, proj_latent_pos, proj_latent_neg = tf.split(embeddings, 3, 0)
                    contrastive_loss = model.actor_network.total_similarity(proj_latent_anch, proj_latent_pos,
                                                                            proj_latent_neg)
                    if eval:
                        logits_pos = tf.matmul(proj_latent_anch, proj_latent_pos,
                                               transpose_b=True) / model.temperature  # + tf.matmul(hidden2, hidden1, transpose_b=True)
                        logits_neg = tf.matmul(proj_latent_anch, proj_latent_neg,
                                               transpose_b=True) / model.temperature  # + tf.matmul(hidden2, hidden1, transpose_b=True)
                        for li, logits in enumerate([logits_pos, logits_neg]):
                            samples = tf.shape(logits)[0]
                            y_pred = tf.cast(tf.argmax(logits), dtype="int32")
                            y_true = tf.cast(tf.range(samples), dtype="int32")
                            const_class_accs[li] += (tf.reduce_sum(
                                tf.cast(tf.math.equal(y_pred, y_true), dtype="int32")) / samples) / len(self.models)
                    model_loss_per_input.append(contrastive_loss)
            possible_gates = len(self.models)
            model_loss_per_input = tf.concat(model_loss_per_input, axis=-1)
            gates, top_k_logits = self.top_k_gating(model_loss_per_input)
            val, idx, count = tf.unique_with_counts(tf.argmax(gates, axis=-1))
            winner_idx = tf.argmax(count)
            winner = val[winner_idx]
            gates = tf.zeros_like(gates, dtype="int64") + winner
            gates = tf.one_hot(gates[:, 0], depth=possible_gates)
            dispatcher = SparseDispatcher(possible_gates, gates)
            dispatched_idxs = []
            dispatched_x = dispatcher.dispatch(x)

            return dispatched_x, \
                   dispatcher.dispatch(xxp) if xxp is not None else dispatched_x, \
                   dispatcher.dispatch(xxn) if xxn is not None else dispatched_x, \
                   dispatched_idxs, \
                   tf.reduce_max(top_k_logits, axis=-1), \
                   const_class_accs, \
                   dispatcher
        else:
            return [x], [xxp], [xxn], [tf.zeros(shape=(len(x), 1)) + 0.9], tf.zeros(
                shape=(len(x), 1)) + 0.99, [0.0, 0.0], None

    @tf.function(experimental_relax_shapes=True)
    def _my_top_k(self, x, k, soft=False):
        """GPU-compatible version of top-k that works for very small constant k.
        Calls argmax repeatedly.
        tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
        seems not to be, so if we use tf.nn.top_k, then both the top_k and its
        gradient go on cpu.  Once this is not an issue, this function becomes
        obsolete and should be replaced by tf.nn.top_k.
        Args:
        x: a 2d Tensor.
        k: a small integer.
        soft: sample or use argmax
        Returns:
        values: a Tensor of shape [batch_size, k]
        indices: a int32 Tensor of shape [batch_size, k]
        """
        if k > 10:
            return tf.math.top_k(x, k)
        values = []
        indices = []
        depth = tf.shape(x)[1]
        for i in range(k):
            if not soft:
                idx = tf.argmax(x, 1)
                values.append(tf.reduce_max(x, 1))
            else:
                dist = tfd.Categorical(logits=x)
                idx = dist.sample()
                values.append(dist.log_prob(idx))
            indices.append(idx)
            if i + 1 < k:
                x += tf.one_hot(idx, depth, -1e9)
        return tf.stack(values, axis=1), tf.cast(tf.stack(indices, axis=1), dtype=tf.int32)

    @tf.function(experimental_relax_shapes=True)
    def _rowwise_unsorted_segment_sum(self, values, indices, n):
        """UnsortedSegmentSum on each row.
        Args:
        values: a `Tensor` with shape `[batch_size, k]`.
        indices: an integer `Tensor` with shape `[batch_size, k]`.
        n: an integer.
        Returns:
        A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
        """
        batch, k = tf.unstack(tf.shape(indices), num=2)
        indices_flat = tf.reshape(indices, [-1]) + tf.cast(tf.divide(tf.range(batch * k), k), dtype=tf.int32) * n
        ret_flat = tf.math.unsorted_segment_sum(tf.reshape(values, [-1]), indices_flat, batch * n)
        return tf.reshape(ret_flat, [batch, n])

    @tf.function(experimental_relax_shapes=True)
    def top_k_gating(self, gating_logits):
        k = 1
        top_logits, top_indices = self._my_top_k(gating_logits, k)
        top_k_logits = tf.slice(top_logits, [0, 0], [-1, k])
        top_k_indices = tf.slice(top_indices, [0, 0], [-1, k])
        top_k_gates = tf.nn.softmax(top_k_logits)
        gates = self._rowwise_unsorted_segment_sum(top_k_gates, top_k_indices, tf.shape(gating_logits)[-1])

        return gates, top_k_logits
