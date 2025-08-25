import torch as tc
from torch import nn

import gymnasium as gym

from ffnn import FFNN
from replay_buffer import ReplayBuffer

from pathlib import Path
import pickle

from typing import Callable
from torch.types import Device

# TODO remove?
from tqdm import trange
import mlflow


class QFFNN(FFNN):
    '''Copy of FFNN whose forward function accepts two inputs instead of one and whose output is squeezed (nicer interface for Q-learning)'''
    def forward(self, state: tc.Tensor, action: tc.Tensor):
        X = tc.concat([state, action], dim=-1)
        return super().forward(X).squeeze()


class SAC(nn.Module):
    '''
    Implementation based on the paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    as well as the OpenAI's spinning-up page "https://spinningup.openai.com/en/latest/algorithms/sac.html"
    '''
    def __init__(
        self,
        q_architecture: list[int | nn.Module],
        policy: nn.Module,
        discount_factor: float,
        temperature: float,
        polyak: float,
        lr_critic: float | tc.Tensor,
        lr_actor: float | tc.Tensor,
        device: Device,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.q1 = QFFNN(q_architecture)
        self.q2 = QFFNN(q_architecture)
        self.pi = policy

        self.optim_q1 = tc.optim.Adam(self.q1.parameters(), lr=lr_critic)
        self.optim_q2 = tc.optim.Adam(self.q2.parameters(), lr=lr_critic)
        self.optim_pi = tc.optim.Adam(self.pi.parameters(), lr=lr_actor)
        self.mse_loss: Callable[[tc.Tensor, tc.Tensor], tc.Tensor] = nn.MSELoss()

        self.q1_target = self.q1.clone()
        self.q2_target = self.q2.clone()

        self.gamma = discount_factor
        self.alpha = temperature
        self.rho = polyak

        self.lr_critic = lr_critic
        self.lr_actor = lr_actor

        self.set_device(device)


    def set_device(self, device: Device):
        self.to(device)
        self.device = device


    def save(self, path_to_dir: str | Path, epoch: int, move_to_cpu: bool = True):
        '''Save model to a single .pth file'''
        # make sure path is of type Path
        if not isinstance(path_to_dir, Path):
            path_to_dir = Path(path_to_dir)

        # make sure path points to a directory
        assert path_to_dir.suffix == "", "path_to_dir must point to a directory!"

        # create folder if necessary
        path_to_dir.mkdir(parents=True, exist_ok=True)

        # move model to cpu TODO check if this slows down training significantly
        if move_to_cpu:
            old_device = self.device
            self.set_device("cpu")

        # put all model data into a single dictionary
        model_dict = {
            'q_architecture': self.q1.architecture,

            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'pi': self.pi.state_dict(),

            'optim_q1': self.optim_q1.state_dict(),
            'optim_q2': self.optim_q2.state_dict(),
            'optim_pi': self.optim_pi.state_dict(),

            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),

            'gamma': self.gamma,
            'alpha': self.alpha,
            'rho': self.rho,

            'lr_critic': self.lr_critic,
            'lr_actor': self.lr_actor,

            'device': self.device,
        }

        # save dictionary to file
        with open(path_to_dir / f"epoch_{epoch}.pth", "wb") as f:
            pickle.dump(model_dict, f)
        
        # move model back to original device
        if move_to_cpu:
            self.set_device(old_device)


    @classmethod
    def load(cls, path_to_dir: str | Path, policy: nn.Module, epoch: int, device: Device):
        '''Load model from previous checkpoint. Since the policy can have any architecture, it cannot be saved and must be supplied'''
        # make sure path is of type Path
        if not isinstance(path_to_dir, Path):
            path_to_dir = Path(path_to_dir)

        # make sure path points to a directory
        assert path_to_dir.suffix == "", "path_to_dir must point to a directory!"

        with open(path_to_dir / f"epoch_{epoch}.pth", "rb") as f:
            model_dict = pickle.load(f)

        policy.load_state_dict(model_dict['pi'])
        out = cls(
            q_architecture = model_dict['q_architecture'],
            policy = policy,
            discount_factor = model_dict['gamma'],
            temperature = model_dict['alpha'],
            polyak = model_dict['rho'],
            lr_critic = model_dict['lr_critic'],
            lr_actor = model_dict['lr_actor'],
            device = model_dict['device'],
        )

        out.q1.load_state_dict(model_dict['q1'])
        out.q2.load_state_dict(model_dict['q2'])

        out.optim_q1.load_state_dict(model_dict['optim_q1'])
        out.optim_q2.load_state_dict(model_dict['optim_q2'])
        out.optim_pi.load_state_dict(model_dict['optim_pi'])

        out.q1_target.load_state_dict(model_dict['q1_target'])
        out.q2_target.load_state_dict(model_dict['q2_target'])

        out.set_device(device)

        return out


    def act(self, state: tc.Tensor) -> tc.Tensor:
        '''Let the policy choose an action'''
        # choose action
        action: tc.Tensor = self.pi(state)[0]

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
        is_warmup: bool = False,
        verbose: bool = False,
    ):
        # heavily inspired by the pseudocode at https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode
        mean_loss_q1 = 0
        mean_loss_q2 = 0
        mean_loss_pi = 0
        mean_ll = 0
        mean_ll_next = 0
        mean_q = 0
        mean_q_new = 0

        for _ in range(n_epochs):
            # sample from replay buffer
            to_tensor = lambda x: tc.stack(x)
            state, action, reward, next_state, terminated, truncated = map(to_tensor, replay_buffer.sample(batch_size))

            # convert terminated to bool
            terminated = terminated.to(tc.bool)

            # --- Critic Training

            # compute target q-value for q-networks
            with tc.no_grad():
                # sample prospective action and its log likelihood
                new_next_action, next_log_likelihood = self.pi(next_state)

                # gauge q-values
                q1_target = self.q1_target(next_state, new_next_action)
                q2_target = self.q2_target(next_state, new_next_action)

                # take those q-values with minimal absolute value instead of the plain minimum to also mitigate negative runoff
                q_target = tc.min(q1_target, q2_target)

                # decide gamma
                gamma = 0 if is_warmup else self.gamma

                # compute modified bellman function
                # if is_warmup:
                #     y = reward / (1 - self.gamma) # roughly aim at the right order of magnitude
                # else:
                #     y = reward + self.gamma * (1 - terminated) * (q_target - self.alpha * next_log_likelihood)
                y = reward.clone()
                y[~terminated] *= (1 - gamma)
                y[~terminated] += gamma * (q_target - self.alpha * next_log_likelihood)


            # perform gradient step for q1-network
            q1 = self.q1(state, action)
            loss_q1 = self.mse_loss(y.detach(), q1)
            assert not (loss_q1.isnan() or loss_q1.isinf())
            self.optim_q1.zero_grad()
            loss_q1.backward()
            self.optim_q1.step()

            # perform gradient step for q2-network
            q2 = self.q2(state, action)
            loss_q2 = self.mse_loss(y.detach(), q2)
            assert not (loss_q2.isnan() or loss_q2.isinf())
            self.optim_q2.zero_grad()
            loss_q2.backward()
            self.optim_q2.step()

            # --- Actor Training

            # sample action and its log likelihood
            new_action, log_likelihood = self.pi(state)

            # compute q-value for newly chosen action
            # with tc.no_grad():
            # self.q1.eval()
            # self.q2.eval()
            q1_new = self.q1(state, new_action)
            q2_new = self.q2(state, new_action)
            q_new = tc.min(q1_new, q2_new)

            # perform gradient step for policy network. Note that the q-value is to be maximized and likelihood is to be minimized, hence the sign
            loss_pi: tc.Tensor = -(q_new - self.alpha * log_likelihood).mean()
            assert not (loss_pi.isnan() or loss_pi.isinf())
            self.optim_pi.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 10) # from Simon, who has it from Pascal
            self.optim_pi.step()

            # self.q1.train()
            # self.q2.train()

            # update target networks
            self.target_update_polyak(is_warmup)

            # track losses
            mean_loss_q1 += float(loss_q1)
            mean_loss_q2 += float(loss_q2)
            mean_loss_pi += float(loss_pi)
            mean_ll += float(log_likelihood.mean())
            mean_ll_next += float(next_log_likelihood.mean())
            mean_q += float(q_target.mean())
            mean_q_new += float(q_new.mean())

        mean_loss_q1 /= n_epochs
        mean_loss_q2 /= n_epochs
        mean_loss_pi /= n_epochs
        mean_ll /= n_epochs
        mean_ll_next /= n_epochs
        mean_q /= n_epochs
        mean_q_new /= n_epochs

        if verbose:
            return mean_loss_q1, mean_loss_q2, mean_loss_pi, mean_ll, mean_ll_next, mean_q, mean_q_new
        else:
            return mean_loss_q1, mean_loss_q2, mean_loss_pi


# small helper function because I need to convert observations (np.ndarray) to suitable states (flattened tc.Tensor) in at least two locations
def _obs_to_state(obs, device: Device):
    '''Converts output from a gym environment to a suitable tensor'''
    state = tc.tensor(obs, dtype=tc.float, device=device) # cast to tensor
    # if not is_batch:
    #     state = state.unsqueeze(0)
    # state = state.flatten(start_dim = 1) # flatten
    return state


def train_agent(
    env: gym.Env | gym.vector.SyncVectorEnv,
    agent: SAC,
    n_episodes: int,
    n_steps_per_episode: int,
    n_train_epochs: int,
    n_warmup_episodes: int,
    batch_size: int,
    replay_buffer: ReplayBuffer | int,
    start_episode: int = 0,
    save_interval: int | None = None,
    save_path: str | Path | None = None,
    show_tqdm: bool = True,
    use_mlflow: bool = True,
):
    '''Training algorithm for a SAC agent in a gym environment with a bounded box action space'''
    # make sure that the action space has the right properties
    # these are limitations of my implementation, not of the general algorithm
    assert isinstance(env.action_space, gym.spaces.Box), "Action space must be a box space!"
    assert env.action_space.bounded_above.all() and env.action_space.bounded_below.all(), "All actions must be bounded!"

    # determine if environment runs in parallel
    # is_batch = isinstance(env, gym.vector.SyncVectorEnv)
    is_batch = hasattr(env, "num_envs")

    # make sure the save params makes sense
    assert (save_interval is None) == (save_path is None), "Parameters 'save_interval' and 'save_path' must either both be given or both be None!"

    # save untrained model
    if start_episode == 0 and save_path is not None:
        agent.save(save_path, 0)

    # initialize replay buffer
    if isinstance(replay_buffer, int):
        replay_buffer = ReplayBuffer(6, maxlen=replay_buffer)
    else:
        assert replay_buffer.n_fields == 6, "Replay buffer must have 6 fields!"

    for i in (trange if show_tqdm else range)(start_episode, n_episodes + start_episode):
        # determine if this is a warmup episode
        is_warmup = (i - start_episode < n_warmup_episodes)

        # reset simulation
        observation, _ = env.reset()
        state = _obs_to_state(observation, agent.device)

        # reset total reward per episode
        total_reward = 0.

        # set agent to eval mode for the duration of the episode
        agent.eval()

        for t in range(n_steps_per_episode):
            # choose action
            if is_warmup:
                action = tc.tensor(env.action_space.sample(), device=agent.device)
            else:
                with tc.no_grad():
                    action = agent.act(state)

            # perform action
            observation, _reward, _terminated, _truncated, info, *_ = env.step(action.cpu().numpy())

            # convert values to tensors (the detaches are quite probably unnecessary)
            state = state.detach()
            action = action.detach()
            next_state = _obs_to_state(observation, agent.device).detach()
            reward = tc.tensor(_reward, dtype=tc.float, device=agent.device).detach()
            terminated = tc.tensor(_terminated, dtype=tc.int, device=agent.device).detach()
            truncated = tc.tensor(_truncated, dtype=tc.int, device=agent.device).detach()

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

            # update state
            state = next_state

        # train agent
        if len(replay_buffer) >= batch_size:
            agent.train()
            l_q1, l_q2, l_pi, *verbose = agent.training_session(replay_buffer, n_train_epochs, batch_size, is_warmup = is_warmup, verbose = use_mlflow)

            # log losses
            if use_mlflow:
                ll, ll_next, q, q_new = verbose
                mlflow.log_metric("loss_q1", l_q1, step=i)
                mlflow.log_metric("loss_q2", l_q2, step=i)
                mlflow.log_metric("loss_pi", l_pi, step=i)
                mlflow.log_metric("log_likelihood", ll, step=i)
                mlflow.log_metric("q", q, step=i)
                mlflow.log_metric("q_new", q_new, step=i)
                mlflow.log_metric("log_likelihood_next", ll_next, step=i)
                mlflow.log_metric("weight_q1", float(agent.q1.layers[0].weight[0,0]), step=i) # type: ignore
                mlflow.log_metric("weight_q2", float(agent.q2.layers[0].weight[0,0]), step=i) # type: ignore
                mlflow.log_metric("weight_q1_target", float(agent.q1_target.layers[0].weight[0,0]), step=i) # type: ignore
                mlflow.log_metric("weight_q2_target", float(agent.q2_target.layers[0].weight[0,0]), step=i) # type: ignore
                mlflow.log_metric("weight_pi", float(agent.pi.ffnn.layers[0].weight[0,0]), step=i) # type: ignore


        # log auxiliary metrics
        if use_mlflow:
            mlflow.log_metric("total_reward", total_reward, step=i)
            mlflow.log_metric("episode_length", t+1, step=i) # type: ignore (t is always bound)
            mlflow.log_metric("mean_reward", total_reward / (t+1), step=i) # type: ignore (t is always bound)
            mlflow.log_metric("buffer_size", len(replay_buffer), step=i)
            mlflow.log_metric("is_warmup", int(is_warmup), step=i)

        # save model at regular intervals and at the very end
        if save_interval is not None and ((i+1) % save_interval == 0 or i+1 == n_episodes + start_episode):
            agent.save(save_path, i+1) # type: ignore (it is guaranteed that save_path is not None)
