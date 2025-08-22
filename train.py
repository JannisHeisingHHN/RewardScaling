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

# these are solely to suppress mlflow's prints. Wouldn't it be nice if there was a flag for those?
import sys
import inspect


#
# FILE USAGE: python train.py <PATH_TO_SETTINGS>
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


def get_step_fn(x: list[float], y: list[float]) -> Callable[[float], float]:
    '''
    Create a step function `f`. For example, if `x = [1, 2]` and `y = [3, 4, 5]`, then `f(-∞) = f(0.9) = 3`,
    `f(1) = f(1.9) = 4` and `f(2) = f(∞) = 5`.

    `y` must have precisely one entry more than `x`.
    '''
    if len(x) + 1 != len(y):
        raise ValueError(f"len(x) must be one less than len(y), but got {len(x)} and {len(y)}.")

    if not (np.diff(x) >= 0).all():
        raise ValueError("x must be monotonically increasing.")

    x_np = np.concatenate([[float("-inf")], x, [float("inf")]])
    y_np = np.concatenate([y, [y[-1]]])

    out = lambda x: float(y_np[(x >= x_np).argmin() - 1])
    out.__doc__ = f"step_fn({x=}, {y=})"

    return out


def get_constant_fn(c: float) -> Callable[[float], float]:
    '''Create a constant function with value `c`.'''
    out = lambda x: c
    out.__doc__ = f"constant_fn({c=})"

    return out


def get_custom_fn(fn_type: str, **params) -> Callable[[float], float]:
    '''Create a custom function of specified type.'''
    match fn_type:
        case "logistic": return get_logistic_fn(**params)
        case "exponential": return get_exponential_fn(**params)
        case "step": return get_step_fn(**params)
        case "constant": return get_constant_fn(**params)
    
    raise ValueError(f"Invalid function type '{fn_type}'.")


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
    custom_reward: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] | None = None,
    truncation_is_termination: bool = False,
    start_episode: int | None = None,
    save_interval: int | None = None,
    save_path: str | Path | None = None,
    show_tqdm: dict[str, Any] | bool = True,
    use_mlflow: bool = True,
):
    '''
    Training algorithm for a Q-learning agent in a gym environment with a discrete action space

    * replay_buffer: Either a replay buffer or the maximum length of the replay buffer.
    * target_update: If float, uses polyak averaging. If int, uses copy with the given periodicity.
    * custom_reward: Maps `(state, action, game_reward, terminated)` to a custom reward. `game_reward` is the reward given by the environment. If set, the custom reward replaces the game reward.
    * truncation_is_termination: If true, truncation (which usually has no effect on training) is treated as termination (which changes the target)
    * show_tqdm: Whether to use the tqm progress bar. May also be a dictionary of arguments, which are then passed to `trange`.
    '''
    # make sure that the action space has the right properties
    assert isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, gym.spaces.MultiDiscrete), "Action space must be discrete!"

    # make sure the save params makes sense
    assert (save_interval is None) == (save_path is None), "Parameters 'save_interval' and 'save_path' must either both be given or both be None!"

    # define reward function
    reward_fn: Callable[[Tensor, Tensor, Tensor, Tensor], Tensor] = (
            (lambda o, a, r, t: r)                          # only use game reward
        if custom_reward is None else
            (lambda o, a, r, t: custom_reward(o, a, r, t))  # use custom reward
    )

    # save untrained model
    if start_episode is None:
        start_episode = 0

        if save_path is not None:
            agent.save(save_path, 0)

    # initialize replay buffer
    if isinstance(replay_buffer, int):
        replay_buffer = ReplayBuffer(replay_buffer, agent.device)

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

        # determine epsilon
        epsilon = epsilon_fn(i)

        # initialise main metrics
        total_reward = 0.
        mean_reward = 0.
        mean_steps = 0.

        # initialise auxiliary metrics
        reward_until_reset = tc.zeros(env.num_envs, device=agent.device)                # accumulated reward per environment
        t_last_reset = tc.zeros(env.num_envs, dtype=tc.int, device=agent.device) - 1    # time of last reset per environment (they're all initially reset "at step -1")
        reset_count = 0                                                                 # reset count for all environments (to divide total_reward & mean_reward by)

        # set agent to eval mode for the duration of the episode
        agent.eval()

        # episode loop
        for t in range(n_steps_per_episode):
            # choose action
            if np.random.rand() < epsilon:
                # random action
                action = env.action_space.sample()
            else:
                # greedy action
                with tc.no_grad():
                    action = agent.act(state)

            # perform action
            observation, _reward, _terminated, _truncated, *_ = env.step(action)

            # fuse terminated and truncated if desired
            if truncation_is_termination:
                _terminated = _terminated | _truncated

            # convert values to tensors
            action = actions_onehot[tc.from_numpy(action)]
            next_state = obs_to_state(observation, agent.device)
            terminated = tc.tensor(_terminated, dtype=tc.bool, device=agent.device)
            truncated = tc.tensor(_truncated, dtype=tc.bool, device=agent.device)
            _reward_tc = tc.tensor(_reward, dtype=tc.float, device=agent.device)

            # compute reward
            reward = reward_fn(state, action, _reward_tc, terminated)

            # accumulate total reward
            reward_until_reset += reward

            # save to buffer
            with tc.no_grad():
                replay_buffer.add_iter(state, action, reward, next_state, terminated, truncated)

            # handle individual environment resets
            for k, (e, done) in enumerate(zip(env.envs, terminated | truncated)):
                if done:
                    # reset environment
                    e.reset()

                    # determine intermediate values
                    new_reward = float(reward_until_reset[k])
                    steps = int(t - t_last_reset[k])

                    # update main metrics
                    total_reward += new_reward
                    mean_reward += new_reward / steps
                    mean_steps += steps


                    # update auxiliary metrics
                    reward_until_reset[k] = 0
                    reset_count += 1
                    t_last_reset[k] = t

            # update state
            state = next_state

        # rectify total and mean reward
        if reset_count == 0: # none of the environments were ever reset -> take average accumulated reward
            total_reward = float(sum(reward_until_reset)) / env.num_envs
            mean_reward = total_reward / n_steps_per_episode
            mean_steps = n_steps_per_episode
        else: # some resets took place -> take average reward accumulated before those resets
            # no division by env.num_envs necessary because reset_count is accumulated across environments
            total_reward /= reset_count
            mean_reward /= reset_count
            mean_steps /= reset_count

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
            mlflow.log_metric("mean_reward", mean_reward, step=i)
            mlflow.log_metric("mean_steps", mean_steps, step=i)
            mlflow.log_metric("reset_count", reset_count / env.num_envs, step=i)
            mlflow.log_metric("buffer_size", len(replay_buffer), step=i)
            mlflow.log_metric("epsilon", epsilon, step=i)

        # save model at regular intervals and at the very end
        if save_interval is not None and ((i+1) % save_interval == 0 or i+1 == n_episodes + start_episode):
            agent.save(save_path, i+1) # type: ignore (it is guaranteed that save_path is not None)


def start_training(settings: dict[str, Any], use_prints: bool = False):
    # load settings
    if use_prints: print("Loading settings...", end="")

    setup: dict[str, Any] = settings['setup']
    params: dict[str, Any] = settings['parameters']

    use_mlflow = setup['use_mlflow']
    mlflow_run_name = setup.get('mlflow_run_name', "norun")
    mlflow_uri = setup.get('mlflow_uri', "nouri")
    mlflow_experiment = setup.get('mlflow_experiment', "noname")
    mlflow_tags = setup.get('mlflow_tags', {})
    show_tqdm = setup.get('show_tqdm', True)

    # define variable parameters
    gamma_fn = get_custom_fn(**params['gamma'])
    epsilon_fn = get_custom_fn(**params['epsilon'])
    lr_fn = get_custom_fn(**params['learning_rate'])

    # the model architecture needs to be evaluated
    architecture_hidden = [eval(layer) for layer in params['architecture_hidden']]

    # handle custom reward
    custom_reward: Callable[[NDArray, NDArray | Tensor, NDArray], NDArray] | None

    if 'custom_reward' not in params:
        # define no custom reward
        custom_reward = None
    else:
        # custom reward function can be given as a string in the settings file
        str_cr: str = params['custom_reward']

        # execute code
        _locals = {}
        exec(str_cr, globals(), _locals)

        # extract custom_reward function
        custom_reward = _locals['custom_reward']

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
        print(" Done!")
        print(f"\tExperiment: {mlflow_experiment}")
        print(f"\tRun: {mlflow_run_name}")
        print(f"\tMlflow: {['off', 'on'][use_mlflow]}")
        print(f"\tDevice: {DEVICE}")


    # initialise environment
    if use_prints: print("Initialising environment and model...", end="")

    discretise: int = params.get('discretise', 0)
    env_generator = (
            (lambda: DiscretiseAction(gym.make(**params['env']), n_actions=discretise))
        if discretise else
            (lambda: gym.make(**params['env']))
    )

    env = gym.vector.SyncVectorEnv([env_generator] * params['num_envs']) # TODO change to gym.vector.make (and probably asynchronous)
    state_size = env.observation_space.shape[-1] # type: ignore
    n_actions = int(env.action_space.nvec[0]) # type: ignore

    # initialize agent
    ModelClass: type[Learner] = eval(params['model_class'])

    start_episode = params.get('start_episode', None)
    if start_episode is not None:
        # torch.load doesn't know that BatchNorm is safe and raises an error by default (to prevent arbitrary code execution)
        with tc.serialization.safe_globals([nn.BatchNorm1d]):
            agent = ModelClass.load(params['save_path'], start_episode, DEVICE)
    else:
        seed = params.get('seed', None)
        agent = ModelClass(
            architecture = [state_size + n_actions] + architecture_hidden + [1],
            n_actions = n_actions,
            use_reward_scaling=params['use_reward_scaling'],
            device = DEVICE,
            seed = seed,
        )

    replay_buffer = ReplayBuffer(params['replay_buffer_size'], DEVICE)

    if use_prints: print(" Done!\nTraining...")

    # determine train arguments
    truncation_is_termination = params.get('truncation_is_termination', False)

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
        'truncation_is_termination': truncation_is_termination,
        'start_episode': start_episode,
        'save_interval': params['save_interval'],
        'save_path': params['save_path'],
        'show_tqdm': show_tqdm,
    }

    # start training
    if use_mlflow is False: # this strange check is necessary because use_mlflow may be an empty dictionary, which would be evaluated as False but should still enable mlflow
        train_agent(
            **train_args,
            use_mlflow = False
        )
    else:
        # remove annoying mlflow prints
        class FilteredStdout:
            def write(self, text: str):
                caller_name: str = inspect.currentframe().f_back.f_globals['__name__'] # type: ignore

                if caller_name.startswith("mlflow"):
                    return

                sys.__stdout__.write(text)

            def flush(self):
                sys.__stdout__.flush()

        sys.stdout = FilteredStdout()

        # make sure to have an mlflow server running
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(mlflow_experiment)

        with mlflow.start_run(run_name=mlflow_run_name):
            mlflow.log_params(params)
            mlflow.set_tags(mlflow_tags)

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
