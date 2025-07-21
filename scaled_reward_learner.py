import numpy as np
import torch as tc
from torch import nn
from torch.nn.functional import one_hot

import gymnasium as gym

from ffnn import FFNN
from replay_buffer import ReplayBuffer

from pathlib import Path
import pickle

from typing import Callable
from numpy.typing import NDArray
from torch.types import Device, Tensor


class QFFNN(FFNN):
    '''Copy of FFNN whose forward function accepts two inputs instead of one and whose output is squeezed (nicer interface for Q-learning)'''
    def forward(self, state: Tensor, action: Tensor):
        X = tc.concat([state, action], dim=-1)
        return super().forward(X).squeeze()
    

    def get_q(self, state: Tensor, actions: Tensor):
        '''
        state has shape (N, D) and actions has shape (M, C), where<br>
        N: number of states (batch size)<br>
        D: number of features in a state<br>
        M: number of actions (must be constant across states)<br>
        C: number of features in an action<br>
        If `state` is one-dimensional, `N` is assumed to be 1, and likewise for `actions` and `C`
        '''
        # add dimensions if necessary
        state = tc.atleast_2d(state) # batch dimension
        actions = actions.view(len(actions), -1) # action feature dimension

        N = len(state)
        M = len(actions)

        # copy states and actions to match with one-another
        S = state.repeat_interleave(M, 0)
        A = actions.repeat(N, 1)

        # get Q-values
        Q: Tensor = self(S, A)

        # reshape to (N, C)
        Q = Q.view(N, -1)

        return Q


    def get_max_q(self, state: Tensor, actions: Tensor):
        # get Q-values
        Q = self.get_q(state, actions)

        # get maximal Q-value per state
        MQ = Q.max(1)[0].squeeze()

        return MQ


    def act(self, state: Tensor, actions: Tensor):
        # get Q-values
        Q = self.get_q(state, actions)

        # get index of best action per state
        idc = Q.argmax(1).squeeze()

        return idc


def _bellmann_std(reward: Tensor, future_reward: Tensor, gamma: float, terminated: Tensor):
    '''Standard Bellman function (status quo)'''
    return reward + gamma * future_reward


def _bellmann_scaled(reward: Tensor, future_reward: Tensor, gamma: float, terminated: Tensor):
    '''Modified Bellman function with appropriate reward scaling (my new method)'''
    out = reward.clone()
    out[~terminated] *= (1 - gamma)
    out[~terminated] += gamma * future_reward

    return out


class ScaledRewardLearner(nn.Module):
    def __init__(
        self,
        architecture: list[int | nn.Module],
        n_actions: int,
        polyak: float,
        use_reward_scaling: bool,
        device: Device,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.q1 = QFFNN(architecture)
        self.q2 = QFFNN(architecture)

        self.optim_q1 = tc.optim.Adam(self.q1.parameters())
        self.optim_q2 = tc.optim.Adam(self.q2.parameters())
        self.mse_loss: Callable[[tc.Tensor, tc.Tensor], tc.Tensor] = nn.MSELoss()

        self.q1_target = self.q1.clone()
        self.q2_target = self.q2.clone()

        self.n_actions = n_actions
        self.actions_onehot = tc.eye(n_actions, dtype=tc.int)

        self.rho = polyak

        # The following if-else morally belongs in training_session as its purpose is to determine how to calculate the Q-functions' targets,
        # but because I'm obsessed with runtime optimization I moved it here
        self.bellman_fn = _bellmann_scaled if use_reward_scaling else _bellmann_std
        self.use_reward_scaling = use_reward_scaling

        self.set_device(device)


    def set_device(self, device: Device):
        self.to(device)
        self.actions_onehot = self.actions_onehot.to(device)
        self.device = device

        return self


    def save(self, path_to_dir: str | Path, epoch: int, move_to_cpu: bool = True):
        '''Save model to a single .pth file'''
        # make sure path is of type Path
        if not isinstance(path_to_dir, Path):
            path_to_dir = Path(path_to_dir)

        # make sure path points to a directory
        assert path_to_dir.suffix == "", "path_to_dir must point to a directory!"

        # create folder if necessary
        path_to_dir.mkdir(parents=True, exist_ok=True)

        # move model to cpu
        if move_to_cpu:
            old_device = self.device
            self.set_device("cpu")

        # put all model data into a single dictionary
        model_dict = {
            'architecture': self.q1.architecture,
            'n_actions': self.n_actions,

            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),

            'optim_q1': self.optim_q1.state_dict(),
            'optim_q2': self.optim_q2.state_dict(),

            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),

            'rho': self.rho,
            'use_reward_scaling': self.use_reward_scaling,

            'device': self.device,
        }

        # save dictionary to file
        with open(path_to_dir / f"epoch_{epoch}.pth", "wb") as f:
            pickle.dump(model_dict, f)
        
        # move model back to original device
        if move_to_cpu:
            self.set_device(old_device)


    @classmethod
    def load(cls, path_to_dir: str | Path, epoch: int, device: Device):
        '''Load model from previous checkpoint'''
        # make sure path is of type Path
        if not isinstance(path_to_dir, Path):
            path_to_dir = Path(path_to_dir)

        # make sure path points to a directory
        assert path_to_dir.suffix == "", "path_to_dir must point to a directory!"

        with open(path_to_dir / f"epoch_{epoch}.pth", "rb") as f:
            model_dict = pickle.load(f)

        out = cls(
            architecture = model_dict['architecture'],
            n_actions = model_dict['n_actions'],
            polyak = model_dict['rho'],
            use_reward_scaling = model_dict['use_reward_scaling'],
            device = model_dict['device'],
        )

        out.q1.load_state_dict(model_dict['q1'])
        out.q2.load_state_dict(model_dict['q2'])

        out.optim_q1.load_state_dict(model_dict['optim_q1'])
        out.optim_q2.load_state_dict(model_dict['optim_q2'])

        out.q1_target.load_state_dict(model_dict['q1_target'])
        out.q2_target.load_state_dict(model_dict['q2_target'])

        out.set_device(device)

        return out


    def act(self, state: tc.Tensor) -> NDArray:
        '''Let the policy choose an action'''
        # choose action
        # action: tc.Tensor = self.pi(state)[0]
        q1 = self.q1.get_q(state, self.actions_onehot)
        q2 = self.q2.get_q(state, self.actions_onehot)

        q = tc.min(q1, q2)
        action = q.argmax(-1).cpu().numpy()

        return action


    def target_update_polyak(self, is_warmup: bool):
        p = 1 if is_warmup else self.rho

        for q, q_target in zip([self.q1, self.q2], [self.q1_target, self.q2_target]):
            s = q.state_dict()
            s_target = q_target.state_dict()
            s_new = {}
            for k in s_target.keys():
                s_new[k] = (1 - p) * s_target[k] + p * s[k]

            q_target.load_state_dict(s_new)


    def training_session(
        self,
        replay_buffer: ReplayBuffer,
        n_epochs: int,
        batch_size: int,
        lr: float,
        gamma: float,
        is_warmup: bool = False,
        verbose: bool = False,
    ):
        mean_loss_q1 = 0
        mean_loss_q2 = 0
        mean_q = 0

        self.optim_q1.param_groups[0]['lr'] = lr
        self.optim_q2.param_groups[0]['lr'] = lr

        for _ in range(n_epochs):
            # sample from replay buffer
            to_tensor = lambda x: tc.stack(x)
            state, action, reward, next_state, terminated, truncated = map(to_tensor, replay_buffer.sample(batch_size))

            # compute target q-value for q-networks
            with tc.no_grad():
                # gauge q-values
                q1_target = self.q1_target.get_max_q(next_state, self.actions_onehot)
                q2_target = self.q2_target.get_max_q(next_state, self.actions_onehot)

                # take those q-values with minimal absolute value instead of the plain minimum to also mitigate negative runoff
                q_target = tc.min(q1_target, q2_target)

                # compute modified bellman function
                y = self.bellman_fn(reward, q_target, gamma, terminated)

            # perform gradient step for q1-network
            q1 = self.q1(state, action)
            loss_q1 = self.mse_loss(y.detach(), q1)
            self.optim_q1.zero_grad()
            loss_q1.backward()
            self.optim_q1.step()

            # perform gradient step for q2-network
            q2 = self.q2(state, action)
            loss_q2 = self.mse_loss(y.detach(), q2)
            self.optim_q2.zero_grad()
            loss_q2.backward()
            self.optim_q2.step()

            # update target networks
            self.target_update_polyak(is_warmup)

            # track losses
            mean_loss_q1 += float(loss_q1)
            mean_loss_q2 += float(loss_q2)
            mean_q += float(q_target.mean())

        mean_loss_q1 /= n_epochs
        mean_loss_q2 /= n_epochs
        mean_q /= n_epochs

        if verbose:
            return mean_loss_q1, mean_loss_q2, mean_q
        else:
            return mean_loss_q1, mean_loss_q2


# small helper function because I need to convert observations (np.ndarray) to suitable states (flattened tc.Tensor) in at least two locations
def _obs_to_state(obs, device: Device):
    '''Converts output from a gym environment to a suitable tensor'''
    state = tc.tensor(obs, dtype=tc.float, device=device)
    return state


def train_agent(
    env: gym.Env | gym.vector.SyncVectorEnv,
    agent: ScaledRewardLearner,
    n_episodes: int,
    n_steps_per_episode: int,
    n_train_epochs: int,
    # n_warmup_episodes: int,
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
    Training algorithm for a Q-learning agent with modified Bellman function in a gym environment with a discrete action space

    * custom_reward: Maps `(observation, action, game_reward)` to a custom reward. `game_reward` is the reward given by the environment. If set, the custom reward replaces the game reward.
    '''
    # make sure that the action space has the right properties
    assert isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, gym.spaces.MultiDiscrete), "Action space must be discrete!"

    # determine if environment runs in parallel
    is_batch = hasattr(env, "num_envs")

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

    for i in (trange if show_tqdm else range)(start_episode, n_episodes + start_episode):
        # reset simulation
        observation, _ = env.reset()
        state = _obs_to_state(observation, agent.device)

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
            observation, _reward, _terminated, _truncated, info, *_ = env.step(action)

            # convert values to tensors (the detaches are quite probably unnecessary TODO remove?)
            state = state.detach()
            action = agent.actions_onehot[action].detach()
            next_state = _obs_to_state(observation, agent.device).detach()
            reward = tc.tensor(reward_fn(observation, action, _reward), dtype=tc.float, device=agent.device).detach()
            terminated = tc.tensor(_terminated, dtype=tc.bool, device=agent.device).detach()
            truncated = tc.tensor(_truncated, dtype=tc.bool, device=agent.device).detach()

            # accumulate total reward
            total_reward += float(tc.mean(reward)) # type: ignore
            assert not reward.isnan().any()

            # save to buffer
            if is_batch:
                replay_buffer.add_iter(state, action, reward, next_state, terminated, truncated)
            else:
                replay_buffer.add(state, action, reward, next_state, terminated, truncated)

            # if done (e.g. because the walker tipped over), stop the episode
            # Alternatively, one could reset the environment and keep the episode going,
            # but that would mess with the total_reward metric
            if terminated.any() or truncated.any(): # TODO: make this work with parallelization and variable episode length
                break

        # train agent
        if len(replay_buffer) >= batch_size:
            lr = lr_fn(i)
            gamma = gamma_fn(i)

            agent.train()
            l_q1, l_q2, *verbose = agent.training_session(
                replay_buffer = replay_buffer,
                n_epochs = n_train_epochs,
                batch_size = batch_size,
                lr = lr,
                gamma = gamma,
                verbose = use_mlflow
            )

            # log training metrics
            if use_mlflow:
                q, *_ = verbose
                mlflow.log_metric("loss_q1", l_q1, step=i)
                mlflow.log_metric("loss_q2", l_q2, step=i)
                mlflow.log_metric("q", q, step=i)
                mlflow.log_metric("weight_q1", float(agent.q1.layers[0].weight[0,0]), step=i) # type: ignore
                mlflow.log_metric("weight_q2", float(agent.q2.layers[0].weight[0,0]), step=i) # type: ignore
                mlflow.log_metric("weight_q1_target", float(agent.q1_target.layers[0].weight[0,0]), step=i) # type: ignore
                mlflow.log_metric("weight_q2_target", float(agent.q2_target.layers[0].weight[0,0]), step=i) # type: ignore
                mlflow.log_metric("lr", lr, step=i)
                mlflow.log_metric("gamma", gamma, step=i)

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
