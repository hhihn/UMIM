import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
######### Configuration files #########
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

#######################################

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False
import tensorflow as tf
import gym
import numpy as np
import pybulletgym.envs
from SAC_agent import Agent
from SAC_utils import ReplayBuffer
from ExperienceBuffer import ExperienceBuffer
from SAC_rla import UMIM
from parallel_gym import make_mp_envs
from copy import deepcopy
import argparse

if __name__ == "__main__":
    print('TensorFlow version: %s' % tf.__version__)
    print('Keras version: %s' % tf.keras.__version__)
    # tf.random.set_seed(seed)
    # np.random.seed(seed)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Mini-Batch Size for SAC Updates')
    parser.add_argument(
        '--hidden_u',
        type=int,
        default=256,
        help='Number of hidden units per layer')
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='Learning rate')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.98,
        help='Discount factor')
    parser.add_argument(
        '--polyak_coef',
        type=float,
        default=0.995,
        help='Polyak Coefficient for running mean updates of target networks')
    parser.add_argument(
        '--train_steps',
        type=int,
        default=1e6,
        help='Total number of Environment Interactions')
    parser.add_argument(
        '--replay_start_size',
        type=int,
        default=1e4,
        help='Number of samples to collect before training starts')
    parser.add_argument(
        '--buffer_size',
        type=int,
        default=1e6,
        help='Size of replay buffer')
    parser.add_argument(
        '--train_interval',
        type=int,
        default=5e3,
        help='Update Model every n steps')
    parser.add_argument(
        '--max_n_models',
        type=int,
        default=2,
        help='Maximum number of models to use')
    parser.add_argument(
        '--save_pref',
        default="",
        help='savename for npy files')

    environments = ['Walker2DPyBulletEnv-v0',
                    'HalfCheetahPyBulletEnv-v0',
                    'AntPyBulletEnv-v0',
                    'InvertedDoublePendulumPyBulletEnv-v0',
                    'HopperPyBulletEnv-v0',
                    'StrikerPyBulletEnv-v0',
                    'ThrowerPyBulletEnv-v0',
                    'HumanoidPyBulletEnv-v0',
                    'PusherPyBulletEnv-v0',
                    'InvertedPendulumPyBulletEnv-v0',
                    'ReacherPyBulletEnv-v0']
    for env in environments:
        print(env)
        e = gym.make(env)
        obs_dim = e.observation_space.shape[0]
        print("Obs dim:", obs_dim, env)
        n_actions = e.action_space.shape[0]
        print("Acts dim:", n_actions, env)

    args = parser.parse_args()
    gamma = args.gamma
    polyak_coef = args.polyak_coef
    train_n_steps = args.train_steps
    batch_size = args.batch_size
    save_pref = args.save_pref
    train_interval = args.train_interval
    n_hidden_units = args.hidden_u
    buffer_size = args.buffer_size
    replay_start_size = args.replay_start_size
    max_num_models = args.max_n_models
    lr = args.lr
    largest_obs_dim = -1
    largest_act_dim = -1
    obs_env = ""
    act_env = ""
    for env_id in environments:
        env = gym.make(env_id)
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]
        if obs_dim > largest_obs_dim:
            largest_obs_dim = obs_dim
            obs_env = env_id
        if n_actions > largest_act_dim:
            largest_act_dim = n_actions
            act_env = env_id

    print("Largest Obs Dim: %d, from Env %s" % (largest_obs_dim, obs_env))
    print("Largest Act Dim: %d, from Env %s" % (largest_act_dim, act_env))

    eval_envs = []
    eval_baselines = []
    old_models = []
    replay_buffers = []
    for _ in range(max_num_models):
        replay_buffer = ExperienceBuffer(
            {'size': buffer_size, 'batch_size': batch_size, 'total_steps': train_n_steps,
             'largest_obs_dim': largest_obs_dim, 'largest_act_dim': largest_act_dim})
        replay_buffers.append(replay_buffer)
    for env_itr, env_id in enumerate(environments):
        min_len = 1e10
        models = []
        replay_buffers = []
        for _ in range(np.minimum(1+env_itr, max_num_models)):
            model = UMIM(save_dir='./',
                         discount=gamma, lr=lr, polyak_coef=polyak_coef, largest_act_dim=largest_act_dim,
                         largest_obs_dim=largest_obs_dim, n_hidden_units=n_hidden_units,
                         task=env_itr, total_episodes=train_n_steps // train_interval)
            # Creating a ReplayBuffer for the training process
            models.append(model)
            replay_buffer = ExperienceBuffer(
                {'size': buffer_size, 'batch_size': batch_size, 'total_steps': train_n_steps,
                 'largest_obs_dim': largest_obs_dim, 'largest_act_dim': largest_act_dim})
            replay_buffers.append(replay_buffer)
        # create an Agent to train / test the model
        agent = Agent(models=models, replay_buffers=replay_buffers,
                      replay_start_size=replay_start_size, batch_size=batch_size,
                      train_n_steps=train_n_steps, largest_act_dim=largest_act_dim, largest_obs_dim=largest_obs_dim,
                      task=env_itr, train_interval=train_interval)
        env = DummyVecEnv([lambda: gym.make(env_id)])  # gym.make(env_id)#
        # Automatically normalize the input features
        train_env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=100000., clip_reward=10000.)
        parallel_eval_envs = make_mp_envs(env_id=env_id, num_env=10, norm_reward=False, start_idx=(1 + env_itr) * 100,
                                          largest_obs_dim=largest_obs_dim)
        eval_envs.append(parallel_eval_envs)

        obs_dim = train_env.observation_space.shape[0]
        print("Obs dim:", obs_dim, env_id)
        n_actions = train_env.action_space.shape[0]
        print("Acts dim:", n_actions, env_id)
        act_lim = train_env.action_space.high
        for model in models:
            model.reset_state()
        agent.reset_state(train_env=train_env, n_actions=n_actions, eval_envs=eval_envs)
        if len(old_models):
            for old_model, new_model in zip(old_models, models):
                new_model.copy_weights(old_models=old_model.get_models(), verbose=False)
        del old_models
        e_reward = agent.train()
        old_models = models
        eval_baselines.append(e_reward)
        del models
        del agent
        print("old models:", len(old_models))

    np.save(arr=eval_baselines,
            file="umim_eval_baselines_noise_2_exp.npy")
