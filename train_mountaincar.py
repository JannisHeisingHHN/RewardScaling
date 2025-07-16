import numpy as np
import torch as tc
from torch import nn

import gymnasium as gym

from scaled_reward_learner import ScaledRewardLearner, train_agent, _obs_to_state
from replay_buffer import ReplayBuffer


# DEVICE = "cuda" if tc.cuda.is_available() else "cpu"
DEVICE = "cpu"
print(f"Device: {DEVICE}")


# decide whether to use mlflow
use_mlflow = True
mlflow_info_tag = "Initial test"


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
    'n_episodes': 57,
    'n_steps_per_episode': 195,
    'n_train_epochs': 5,
    'n_warmup_episodes': 10,
    'batch_size': 1013,

    # rates & coefficients
    'discount_factor': 0.95,
    'polyak_coefficient': 0.005,
    'lr': 3e-4,

    # replay buffer
    'replay_buffer_size': 100_000,

    # save parameters
    'start_episode': 0,
    'save_interval': 10,
    'save_path': "weights/srl_mountaincar",
}


# Initialise everything for training
env: gym.vector.SyncVectorEnv = gym.make_vec(**params['env'], vectorization_mode="sync") # type: ignore

state_size = env.observation_space.shape[-1] # type: ignore
action_size = int(env.action_space.nvec[0]) # type: ignore

# policy = ConstantPolicy(tc.tensor([0], device=DEVICE))


agent = ScaledRewardLearner(
    architecture = [state_size + action_size] + params['architecture_hidden'] + [1],
    n_actions = int(env.action_space.nvec[0]), # type: ignore
    discount_factor = params['discount_factor'],
    polyak = params['polyak_coefficient'],
    lr = params['lr'],
    device = DEVICE,
)

replay_buffer = ReplayBuffer(6, params['replay_buffer_size'])

if use_mlflow:
    import mlflow

    # make sure to have an mlflow server running with "mlflow server --host 127.0.0.1 --port 8080"
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("SRL_MountainCar-v0")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("Info", mlflow_info_tag) # !!! UPDATE !!!

        try:
            train_agent(
                env,
                agent,
                n_episodes = params['n_episodes'],
                n_steps_per_episode = params['n_steps_per_episode'],
                n_train_epochs = params['n_train_epochs'],
                n_warmup_episodes = params['n_warmup_episodes'],
                batch_size = params['batch_size'],
                # replay_buffer = params['replay_buffer_size'],
                replay_buffer = replay_buffer,
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
        env,
        agent,
        n_episodes = params['n_episodes'],
        n_steps_per_episode = params['n_steps_per_episode'],
        n_train_epochs = params['n_train_epochs'],
        n_warmup_episodes = params['n_warmup_episodes'],
        batch_size = params['batch_size'],
        # replay_buffer = params['replay_buffer_size'],
        replay_buffer = replay_buffer,
        start_episode = params['start_episode'],
        save_interval = params['save_interval'],
        save_path = params['save_path'],
        use_mlflow = False,
    )
