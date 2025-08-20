import numpy as np
import torch as tc
from torch import nn

import gymnasium as gym

from agents import *
from envs import *

from typing import Callable, Any
from torch.types import Device
from torch import Tensor
from numpy.typing import NDArray

from pathlib import Path
import toml

from tqdm import trange
import mlflow


#
# FILE USAGE: python train_mountaincar.py <PATH_TO_SETTINGS>
#


def get_logistic_fn(start: float, end: float, midpoint: float, rate: float) -> Callable[[float], float]:
    '''
    Create a logistic function of the form `f(x) = start + (end - start) / (1 + exp(-rate*(x - midpoint)))`.
    This function satisfies `f(-∞) = start`, `f(∞) = end`. If `midpoint` is reasonably large, then `f(0) ≈ start`.
    '''
    out = lambda x: start + (end - start) / (1 + np.exp(-rate*(x - midpoint)))
    out.__doc__ = f"logistic_fn({start=}, {end=}, {midpoint=}, {rate=})"
    return out


def get_exponential_fn(start: float, factor: float) -> Callable[[float], float]:
    '''
    Create an exponential function of the form `f(x) = start * factor^x`.
    '''
    out = lambda x: start * (factor**x)
    out.__doc__ = f"exponential_fn({start=}, {factor=})"
    return out


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
    target_update: float | int,
    custom_reward: Callable[[NDArray, NDArray | Tensor, NDArray], NDArray] | None = None,
    start_episode: int = 0,
    save_interval: int | None = None,
    save_path: str | Path | None = None,
    show_tqdm: dict[str, Any] | bool = True,
    use_mlflow: bool = True,
):
    '''
    Training algorithm for a Q-learning agent in a gym environment with a discrete action space

    * target_update: If float, uses polyak averaging. If int, uses copy with the given periodicity.
    * custom_reward: Maps `(observation, action, game_reward)` to a custom reward. `game_reward` is the reward given by the environment. If set, the custom reward replaces the game reward.
    * show_tqdm: Whether to use the tqm progress bar. May also be a dictionary of arguments, which are then passed to `trange`.
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

    # initialize replay buffer
    if isinstance(replay_buffer, int):
        replay_buffer = ReplayBuffer(6, maxlen=replay_buffer)
    else:
        assert replay_buffer.n_fields == 6, "Replay buffer must have 6 fields!"

    # create action lookup
    n_actions = int(env.action_space.nvec[0]) # type: ignore
    actions_onehot = tc.eye(n_actions, device=agent.device)

    # select target update policy
    use_target_copy = isinstance(target_update, int)

    # select iterator
    start = start_episode
    stop = n_episodes + start_episode

    if isinstance(show_tqdm, dict):
        iterator = trange(start, stop, **show_tqdm)
    elif show_tqdm:
        iterator = trange(start, stop)
    else:
        iterator = range(start, stop)

    # run training
    for i in iterator:
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
                    action = agent.act(state)

            # perform action
            observation, _reward, _terminated, _truncated, *_ = env.step(action)

            # convert values to tensors (the detaches are quite probably unnecessary TODO remove?)
            state = state.detach()
            action = actions_onehot[tc.from_numpy(action)].detach()
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

            # update state
            state = next_state

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

            # update target network
            if use_target_copy: # hard update
                if i % target_update == 0:
                    agent.target_update_copy()
            else: # soft update
                agent.target_update_polyak(target_update)

            # log training metrics
            if use_mlflow:
                mlflow.log_metric("lr", lr, step=i)
                mlflow.log_metric("gamma", gamma, step=i)
                mlflow.log_metrics(train_metrics, step=i)
                mlflow.log_metrics(agent.mlflow_get_sample_weights(), step=i)

        # log auxiliary metrics
        if use_mlflow:
            mlflow.log_metric("total_reward", total_reward, step=i)
            mlflow.log_metric("episode_length", t+1, step=i) # type: ignore (t is always bound)
            mlflow.log_metric("mean_reward", total_reward / (t+1), step=i) # type: ignore (t is always bound)
            mlflow.log_metric("buffer_size", len(replay_buffer), step=i)
            mlflow.log_metric("epsilon", epsilon, step=i) # type: ignore (epsilon is always bound)

        # save model at regular intervals and at the very end
        if save_interval is not None and ((i+1) % save_interval == 0 or i+1 == n_episodes + start_episode):
            agent.save(save_path, i+1) # type: ignore (it is guaranteed that save_path is not None)


def start_training(settings: dict[str, Any], use_prints: bool = False):
    # load settings
    if use_prints: print("Loading settings...", end="")

    use_mlflow = settings['setup']['use_mlflow']
    mlflow_run_name = settings['setup'].get('mlflow_run_name', "norun")
    mlflow_uri = settings['setup'].get('mlflow_uri', "nouri")
    mlflow_experiment = settings['setup'].get('mlflow_experiment', "noname")
    show_tqdm = settings['setup'].get('show_tqdm', True)

    params: dict[str, Any] = settings['parameters']
    discretise: int = params.get('discretise', 0)

    # define variable parameters
    gamma_fn = get_logistic_fn(**params['gamma'])
    epsilon_fn = get_logistic_fn(**params['epsilon'])
    lr_fn = get_exponential_fn(**params['learning_rate'])

    # the model architecture needs to be evaluated
    architecture_hidden = [eval(layer) for layer in params['architecture_hidden']]

    # handle custom reward
    custom_reward: Callable[[NDArray, NDArray | Tensor, NDArray], NDArray] | None
    custom_reward_doc: str

    if 'custom_reward' not in params:
        # define no custom reward
        custom_reward = None
        custom_reward_doc = ""
    else:
        # custom reward function can be given as a string in the settings file
        str_cr: str = params['custom_reward']

        # execute code
        _locals = {}
        exec(str_cr, _locals)

        # extract custom_reward function
        custom_reward = _locals['custom_reward']

        # get custom_reward documentation (if given)
        custom_reward_doc = params.get('custom_reward_doc', str_cr)

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
        # default device is cpu
        DEVICE = "cpu"

    # register custom environment if needed
    if 'registration' in params:
        gym.register(id = params['env']['id'], **params['registration'])

    if use_prints:
        print("Done!")
        print(f"\tExperiment: {mlflow_experiment}")
        print(f"\tRun: {mlflow_run_name}")
        print(f"\tMlflow: {['off', 'on'][use_mlflow]}")
        print(f"\tDevice: {DEVICE}")


    # initialise environment
    if use_prints: print("Initialising environment and model...", end="")

    env_generator = (
            (lambda: DiscretiseAction(gym.make(**params['env']), n_actions=discretise))
        if discretise else
            (lambda: gym.make(**params['env']))
    )

    env = gym.vector.SyncVectorEnv([env_generator] * params['num_envs'])
    state_size = env.observation_space.shape[-1] # type: ignore
    n_actions = int(env.action_space.nvec[0]) # type: ignore

    # initialize agent
    ModelClass: type[Learner] = eval(params['model_class'])

    if (se := params['start_episode']) > 0:
        # torch.load doesn't know that BatchNorm is safe and raises an error by default (to prevent arbitrary code execution)
        with tc.serialization.safe_globals([nn.BatchNorm1d]):
            agent = ModelClass.load(params['save_path'], se, DEVICE)
    else:
        agent = ModelClass(
            architecture = [state_size + n_actions] + architecture_hidden + [1],
            n_actions = n_actions,
            use_reward_scaling=params['use_reward_scaling'],
            device = DEVICE,
        )

    replay_buffer = ReplayBuffer(6, params['replay_buffer_size'])

    if use_prints: print(" Done!\nTraining...")

    # determine train arguments
    train_args = {
        'env': env,
        'agent': agent,
        'n_episodes': params['n_episodes'],
        'n_steps_per_episode': params['n_steps_per_episode'],
        'n_train_epochs': params['n_train_epochs'],
        'batch_size': params['batch_size'],
        'replay_buffer': replay_buffer,
        'lr_fn': lr_fn,
        'epsilon_fn': epsilon_fn,
        'gamma_fn': gamma_fn,
        'target_update': params['target_update'],
        'custom_reward': custom_reward,
        'start_episode': params['start_episode'],
        'save_interval': params['save_interval'],
        'save_path': params['save_path'],
        'show_tqdm': show_tqdm,
    }

    # start training
    if use_mlflow == False: # this strange check is necessary because use_mlflow may be an empty dictionary, which should still enable mlflow
        train_agent(
            **train_args,
            use_mlflow = False
        )
    else:
        # make sure to have an mlflow server running with "mlflow server --host 127.0.0.1 --port 8080"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name=mlflow_run_name):
            mlflow.log_params(params)

            try:
                train_agent(
                    **train_args,
                    use_mlflow = True
                )
            except KeyboardInterrupt:
                mlflow.set_tag("Note", "Manual interruption")
                pass
            except AssertionError:
                mlflow.set_tag("Note", "Interruption by assertion error")
    
    if use_prints: print(" Done!")



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        # settings file was not provided
        raise TypeError(f"Must supply a path to a settings.toml file! Usage: python {sys.argv[0]} <PATH_TO_SETTINGS>")

    # try to load settings
    path_to_settings = sys.argv[1]
    try:
        settings = toml.load(path_to_settings)
    except FileNotFoundError:
        raise ValueError(f"Settings file '{path_to_settings}' was not found!")

    # run training
    start_training(settings, use_prints=True)
