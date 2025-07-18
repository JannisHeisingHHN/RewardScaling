import numpy as np
import torch as tc
from torch import nn

import gymnasium as gym

from scaled_reward_learner import ScaledRewardLearner, train_agent, _obs_to_state
from replay_buffer import ReplayBuffer

from torch.types import Tensor
from numpy.typing import NDArray


DEVICE = "cuda" if tc.cuda.is_available() else "cpu"
# DEVICE = "cpu"
print(f"Device: {DEVICE}")


# decide whether to use mlflow
use_mlflow = False
mlflow_info_tag = "Training continuation test"

# mlflow_uri = "http://127.0.0.1:8080"
# mlflow_experiment = "SRL_MountainCar-v0"

mlflow_uri = "http://10.30.20.11:5000"
mlflow_experiment = "jheis_SRL_MountainCar-v0"

if use_mlflow:
    print(f"MLFlow: {mlflow_uri} | {mlflow_experiment} | {mlflow_info_tag}")
else:
    print("MLFlow: False")


# set hyperparameters
params = {
    # environment
    'env': {
        'id': "MountainCar-v0",
        'num_envs': 11,
    },

    # model architecture (excluding input & output size since these are determined by the environment)
    'architecture_hidden': [256, nn.BatchNorm1d(256), 256],

    # training lengths
    'n_episodes': 1500,
    'n_steps_per_episode': 195,
    'n_train_epochs': 5, # training cycles per epoch
    'batch_size': 1013,

    # rates & coefficients
    'polyak_coefficient': 0.005,

    # replay buffer
    'replay_buffer_size': 100_000,

    # save parameters
    'start_episode': 100,
    'save_interval': 50,
    'save_path': "weights/srl_mountaincar",
}


# Initialise agent
env: gym.vector.SyncVectorEnv = gym.make_vec(**params['env'], vectorization_mode="sync") # type: ignore

if (se := params['start_episode']) > 0:
    print(f"Loading from epoch {se}")
    agent = ScaledRewardLearner.load(params['save_path'], se, DEVICE)
else:
    state_size = env.observation_space.shape[-1] # type: ignore
    action_size = int(env.action_space.nvec[0]) # type: ignore

    agent = ScaledRewardLearner(
        architecture = [state_size + action_size] + params['architecture_hidden'] + [1],
        n_actions = int(env.action_space.nvec[0]), # type: ignore
        polyak = params['polyak_coefficient'],
        device = DEVICE,
    )

replay_buffer = ReplayBuffer(6, params['replay_buffer_size'])

# the docstrings are used as the parameter value for mlflow
def lr_fn(epoch: int):
    '''lr = 3e-4 * (1 - 1e-6)**epoch'''
    lr = 3e-4 * (1 - 1e-6)**epoch

    return float(lr)


def epsilon_fn(epoch: int):
    '''eps = (eps_start - eps_end) * np.exp(-epoch * eps_v) + eps_end'''
    eps_start = 0.9
    eps_end = 0.05
    eps_v = 0.005 # velocity
    eps = (eps_start - eps_end) * np.exp(-epoch * eps_v) + eps_end

    return float(eps)


def gamma_fn(epoch: int):
    '''gamma = (gamma_start - gamma_end) * np.exp(-epoch * gamma_v) + gamma_end'''
    gamma_start = 0.0
    gamma_end = 0.99
    gamma_v = 0.002 # velocity
    gamma = (gamma_start - gamma_end) * np.exp(-epoch * gamma_v) + gamma_end

    return float(gamma)


def custom_reward(state: NDArray, action: NDArray | Tensor, reward: NDArray):
    '''reward = x_pos + flag_bonus'''
    x_pos = state[:, 0] # x-position of cart
    flag_bonus = 2 * (x_pos >= 0.5) # reward for reaching the flag (essentially the game reward), scaled so it's always greater than the x-position
    reward = x_pos + flag_bonus

    return reward


if use_mlflow:
    import mlflow

    # make sure to have an mlflow server running with "mlflow server --host 127.0.0.1 --port 8080"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("lr_fn", lr_fn.__doc__)
        mlflow.log_param("epsilon_fn", epsilon_fn.__doc__)
        mlflow.log_param("gamma_fn", gamma_fn.__doc__)
        mlflow.log_param("custom_reward", custom_reward.__doc__)
        mlflow.set_tag("Info", mlflow_info_tag)

        try:
            train_agent(
                env = env,
                agent = agent,
                n_episodes = params['n_episodes'],
                n_steps_per_episode = params['n_steps_per_episode'],
                n_train_epochs = params['n_train_epochs'],
                batch_size = params['batch_size'],
                replay_buffer = replay_buffer,
                lr_fn = lr_fn,
                epsilon_fn = epsilon_fn,
                gamma_fn = gamma_fn,
                custom_reward = custom_reward,
                start_episode = params['start_episode'],
                save_interval = params['save_interval'],
                save_path = params['save_path'],
            )
        except KeyboardInterrupt:
            mlflow.set_tag("Note", "Manual interruption")
            pass
        except AssertionError:
            mlflow.set_tag("Note", "Interruption by assertion error")
else:
    train_agent(
        env = env,
        agent = agent,
        n_episodes = params['n_episodes'],
        n_steps_per_episode = params['n_steps_per_episode'],
        n_train_epochs = params['n_train_epochs'],
        batch_size = params['batch_size'],
        replay_buffer = replay_buffer,
        lr_fn = lr_fn,
        epsilon_fn = epsilon_fn,
        gamma_fn = gamma_fn,
        custom_reward = custom_reward,
        start_episode = params['start_episode'],
        save_interval = params['save_interval'],
        save_path = params['save_path'],
        use_mlflow = False,
    )
