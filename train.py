import numpy as np
import torch as tc
from torch import nn

import gymnasium as gym

from agents import *

from typing import Callable
from torch.types import Device
from torch import Tensor
from numpy.typing import NDArray

from pathlib import Path
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

def custom_reward(state: NDArray, action: NDArray | Tensor, reward: NDArray): # TODO: add to settings.toml?
    '''reward = ((y_pos + 1) / 2)^2 + flag_bonus'''
    x_pos = state[:, 0] # x-position of cart
    y_pos = np.sin(3 * x_pos) # y-position of cart
    pos_reward = ((y_pos + 1) / 2)**2 # give reward based on cart position
    flag_bonus = 2 * (x_pos >= 0.5) # reward for reaching the flag (essentially the game reward), scaled so it's always greater than the position reward
    reward = pos_reward + flag_bonus

    return reward


def obs_to_state(obs, device: Device):
    '''Converts output from a gym environment to a suitable tensor'''
    state = tc.tensor(obs, dtype=tc.float, device=device)
    return state


def train_agent(
    env: gym.vector.SyncVectorEnv,
    agent: Learner,
    n_episodes: int,
    n_steps_per_episode: int,
    n_train_epochs: int,
    batch_size: int,
    replay_buffer: ReplayBuffer | int,
    lr_fn: Callable[[int], float],
    epsilon_fn: Callable[[int], float],
    gamma_fn: Callable[[int], float],
    custom_reward: Callable[[NDArray, NDArray | Tensor, NDArray], NDArray] | None = None,
    start_episode: int = 0,
    save_interval: int | None = None,
    save_path: str | Path | None = None,
    show_tqdm: bool = True,
    use_mlflow: bool = True,
):
    '''
    Training algorithm for a Q-learning agent in a gym environment with a discrete action space

    * custom_reward: Maps `(observation, action, game_reward)` to a custom reward. `game_reward` is the reward given by the environment. If set, the custom reward replaces the game reward.
    '''
    # make sure that the action space has the right properties
    assert isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, gym.spaces.MultiDiscrete), "Action space must be discrete!"

    # make sure the save params makes sense
    assert (save_interval is None) == (save_path is None), "Parameters 'save_interval' and 'save_path' must either both be given or both be None!"

    # define reward function
    reward_fn = (
            (lambda o, a, r: r) # only use game reward
        if custom_reward is None else
            (lambda o, a, r: custom_reward(o, a, r)) # use custom reward
    )

    # save untrained model
    if start_episode == 0 and save_path is not None:
        agent.save(save_path, 0)

    # import optional libraries
    if show_tqdm: from tqdm import trange
    if use_mlflow: import mlflow

    # initialize replay buffer
    if isinstance(replay_buffer, int):
        replay_buffer = ReplayBuffer(6, maxlen=replay_buffer)
    else:
        assert replay_buffer.n_fields == 6, "Replay buffer must have 6 fields!"

    # create action lookup
    n_actions = int(env.action_space.nvec[0]) # type: ignore
    actions_onehot = tc.eye(n_actions, dtype=tc.int, device=agent.device)

    for i in (trange if show_tqdm else range)(start_episode, n_episodes + start_episode):
        # reset simulation
        observation, _ = env.reset()
        state = obs_to_state(observation, agent.device)

        # reset total reward per episode
        total_reward = 0.

        # set agent to eval mode for the duration of the episode
        agent.eval()

        for t in range(n_steps_per_episode):
            # choose action
            epsilon = epsilon_fn(i)
            if np.random.rand() < epsilon:
                # random action
                action = env.action_space.sample()
            else:
                # greedy action
                with tc.no_grad():
                    action = agent.act(state, actions_onehot)

            # perform action
            observation, _reward, _terminated, _truncated, info, *_ = env.step(action)

            # convert values to tensors (the detaches are quite probably unnecessary TODO remove?)
            state = state.detach()
            action = actions_onehot[action].detach()
            next_state = obs_to_state(observation, agent.device).detach()
            reward = tc.tensor(reward_fn(observation, action, _reward), dtype=tc.float, device=agent.device).detach()
            terminated = tc.tensor(_terminated, dtype=tc.bool, device=agent.device).detach()
            truncated = tc.tensor(_truncated, dtype=tc.bool, device=agent.device).detach()

            # accumulate total reward
            total_reward += float(tc.mean(reward)) # type: ignore

            # save to buffer
            replay_buffer.add_iter(state, action, reward, next_state, terminated, truncated)

            # if done, stop the episode
            # Alternatively, one could reset the environment and keep the episode going,
            # but that would mess with the total_reward metric
            if terminated.any() or truncated.any(): # TODO: make this work with parallelization and variable episode length
                break

        # train agent
        if len(replay_buffer) >= batch_size:
            lr = lr_fn(i)
            gamma = gamma_fn(i)

            agent.train()
            train_metrics = agent.training_session(
                replay_buffer = replay_buffer,
                n_epochs = n_train_epochs,
                batch_size = batch_size,
                lr = lr,
                gamma = gamma,
            )

            # log training metrics
            if use_mlflow:
                mlflow.log_metric("lr", lr, step=i)
                mlflow.log_metric("gamma", gamma, step=i)
                mlflow.log_metrics(train_metrics, step=i)
                mlflow.log_metrics(agent.mlflow_get_sample_weights(), step=i)

        # log auxiliary metrics
        if use_mlflow:
            mlflow.log_metric("total_reward", total_reward, step=i)
            mlflow.log_metric("episode_length", t+1, step=i)
            mlflow.log_metric("mean_reward", total_reward / (t+1), step=i)
            mlflow.log_metric("buffer_size", len(replay_buffer), step=i)
            mlflow.log_metric("epsilon", epsilon, step=i)

        # save model at regular intervals and at the very end
        if save_interval is not None and ((i+1) % save_interval == 0 or i+1 == n_episodes + start_episode):
            agent.save(save_path, i+1) # type: ignore (it is guaranteed that save_path is not None)

        # update state
        state = next_state


def main(settings: dict):
    # load settings
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

    # determine device
    devices: str | list[str] = settings['setup']['device']

    if isinstance(devices, str):
        # single device given
        devices = [devices]

    # choose the first device that is available
    for d in devices:
        try:
            _ = tc.tensor(0, device=d) # this would throw the error
            DEVICE = str(d) # device deemed available
            break
        except RuntimeError:
            pass
    else:
        # default device: cpu
        DEVICE = "cpu"

    # print info
    print(f"Device: {DEVICE}")
    if use_mlflow:
        print(f"MLFlow: {mlflow_uri} | {mlflow_experiment} | {mlflow_run_name}")
    else:
        print("MLFlow: False")

    # initialize environment
    env = gym.vector.SyncVectorEnv([lambda: gym.make(**params['env'])] * params['num_envs'])
    state_size = env.observation_space.shape[-1] # type: ignore
    n_actions = int(env.action_space.nvec[0]) # type: ignore

    # initialize agent
    ModelClass: type[Learner] = eval(params['model_class'])

    if (se := params['start_episode']) > 0:
        print(f"Loading from epoch {se}")

        # torch.load doesn't know that BatchNorm is safe and raises an error by default (to prevent arbitrary code execution)
        with tc.serialization.safe_globals([nn.BatchNorm1d]):
            agent = ModelClass.load(params['save_path'], se, DEVICE)
    else:

        agent = ModelClass(
            architecture = [state_size + n_actions] + architecture_hidden + [1],
            n_actions = n_actions,
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
