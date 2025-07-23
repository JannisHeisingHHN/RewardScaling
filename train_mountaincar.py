import numpy as np
import torch as tc
from torch import nn

import gymnasium as gym

from scaled_reward_learner import ScaledRewardLearner, train_agent, _obs_to_state
from replay_buffer import ReplayBuffer

from typing import Callable
from torch import Tensor
from numpy.typing import NDArray

import toml

#
# FILE USAGE: python train_mountaincar.py <PATH_TO_SETTINGS>
#


# Set variable parameters. The docstrings are used as the parameter value for mlflow
def get_logistic_fn(start: float, end: float, midpoint: float, rate: float) -> Callable[[float], float]:
    '''
    Create a logistic function of the form `f(x) = start + (end - start) / (1 + exp(-rate*(x - midpoint)))`<br>.
    This function satisfies `f(-∞) = start`, `f(∞) = end`. If `midpoint` is reasonably large, then `f(0) ≈ start`.
    '''
    out = lambda x: start + (end - start) / (1 + np.exp(-rate*(x - midpoint)))
    out.__doc__ = f"logistic_fn({start=}, {end=}, {midpoint=}, {rate=})"
    return out

def lr_fn(epoch: int): # TODO: add to settings.toml
    '''lr = 3e-4 * (1 - 3e-4)**epoch'''
    lr = 3e-4 * (1 - 3e-4)**epoch

    return float(lr)

def custom_reward(state: NDArray, action: NDArray | Tensor, reward: NDArray): # TODO: addto settings.toml?
    '''reward = y_pos + flag_bonus'''
    x_pos = state[:, 0] # x-position of cart
    pos_reward = np.sin(3 * x_pos) # give reward based on achieved height
    flag_bonus = 2 * (x_pos >= 0.5) # reward for reaching the flag (essentially the game reward), scaled so it's always greater than the position reward
    reward = pos_reward + flag_bonus

    return reward


def main(settings: dict):
    # load settings
    DEVICE = settings['setup']['device']

    use_mlflow = settings['setup']['use_mlflow']
    mlflow_run_name = settings['setup']['mlflow_run_name']
    mlflow_uri = settings['setup']['mlflow_uri']
    mlflow_experiment = settings['setup']['mlflow_experiment']

    params = settings['parameters']

    # define variable parameters
    gamma_fn = get_logistic_fn(**params['gamma'])
    epsilon_fn = get_logistic_fn(**params['epsilon'])

    # the model architecture needs to be evaluated
    architecture_hidden = [eval(layer) for layer in params['architecture_hidden']]

    # print info
    print(f"Device: {DEVICE}")
    if use_mlflow:
        print(f"MLFlow: {mlflow_uri} | {mlflow_experiment} | {mlflow_run_name}")
    else:
        print("MLFlow: False")

    # initialize agent
    env: gym.vector.SyncVectorEnv = gym.make_vec(**params['env'], vectorization_mode="sync") # type: ignore

    if (se := params['start_episode']) > 0:
        print(f"Loading from epoch {se}")

        # torch.load doesn't know that BatchNorm is safe and raises an error by default (to prevent arbitrary code execution)
        with tc.serialization.safe_globals([nn.BatchNorm1d]):
            agent = ScaledRewardLearner.load(params['save_path'], se, DEVICE)
    else:
        state_size = env.observation_space.shape[-1] # type: ignore
        action_size = int(env.action_space.nvec[0]) # type: ignore

        agent = ScaledRewardLearner(
            architecture = [state_size + action_size] + architecture_hidden + [1],
            n_actions = int(env.action_space.nvec[0]), # type: ignore
            polyak = params['polyak_coefficient'],
            use_reward_scaling=params['use_reward_scaling'],
            device = DEVICE,
        )

    replay_buffer = ReplayBuffer(6, params['replay_buffer_size'])

    # start training
    if use_mlflow:
        import mlflow

        # make sure to have an mlflow server running with "mlflow server --host 127.0.0.1 --port 8080"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name=mlflow_run_name):
            mlflow.log_params(params)
            mlflow.log_param("lr_fn", lr_fn.__doc__)
            mlflow.log_param("epsilon_fn", epsilon_fn.__doc__)
            mlflow.log_param("gamma_fn", gamma_fn.__doc__)
            mlflow.log_param("custom_reward", custom_reward.__doc__)

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


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        # settings file was not provided
        print(f"Must supply a path to a settings.toml file! Usage: python {sys.argv[0]} <PATH_TO_SETTINGS>")
        input("Press enter to exit")
        exit()

    # try to load settings
    path_to_settings = sys.argv[1]
    try:
        settings = toml.load(path_to_settings)
    except FileNotFoundError:
        print(f"Settings file '{path_to_settings}' was not found")
        input("Press enter to exit")
        exit()

    # run training
    main(settings)
