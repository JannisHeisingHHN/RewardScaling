import torch as tc
from torch import nn

from .qffnn import QFFNN
from .learner import Learner
from .replay_buffer import ReplayBuffer

from typing import Callable
from numpy.typing import NDArray
from torch.types import Device
from torch import Tensor


def _bellmann_std(reward: Tensor, future_reward: Tensor, gamma: float, terminated: Tensor):
    '''Standard Bellman function (status quo)'''
    out = reward.clone()
    out[~terminated] += gamma * future_reward

    return out


def _bellmann_scaled(reward: Tensor, future_reward: Tensor, gamma: float, terminated: Tensor):
    '''Modified Bellman function with appropriate reward scaling (my new method)'''
    out = reward.clone()
    out[~terminated] *= (1 - gamma)
    out[~terminated] += gamma * future_reward

    return out


class SingleLearner(Learner):
    def __init__(
        self,
        architecture: list[int | nn.Module],
        n_actions: int,
        use_reward_scaling: bool,
        device: Device,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.q1 = QFFNN(architecture)

        self.optim_q1 = tc.optim.Adam(self.q1.parameters(), eps=0.000001)
        # self.loss_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor] = nn.MSELoss()
        self.loss_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor] = nn.SmoothL1Loss() # TODO test if this works better

        self.q1_target = self.q1.clone()

        self.n_actions = n_actions
        self.actions_onehot = tc.eye(n_actions)

        # The following if-else morally belongs in training_session (where self.bellman_fn is used) as its purpose is to determine
        # how to calculate the Q-functions' targets, but because I'm obsessed with runtime optimization I moved it here
        self.bellman_fn = _bellmann_scaled if use_reward_scaling else _bellmann_std
        self.use_reward_scaling = use_reward_scaling

        self.set_device(device)


    def set_device(self, device: Device):
        self.actions_onehot = self.actions_onehot.to(device)
        return super().set_device(device)


    def to_dict(self):
        # put all model data into a single dictionary
        model_dict = {
            'architecture': self.q1.architecture,
            'n_actions': self.n_actions,

            'q1': self.q1.state_dict(),

            'optim_q1': self.optim_q1.state_dict(),

            'q1_target': self.q1_target.state_dict(),

            'use_reward_scaling': self.use_reward_scaling,
        }

        return model_dict


    @classmethod
    def from_dict(cls, model_dict, device):
        '''Load model from previous checkpoint'''
        out = cls(
            architecture = model_dict['architecture'],
            n_actions = model_dict['n_actions'],
            use_reward_scaling = model_dict['use_reward_scaling'],
            device = device,
        )

        out.q1.load_state_dict(model_dict['q1'])

        out.optim_q1.load_state_dict(model_dict['optim_q1'])

        out.q1_target.load_state_dict(model_dict['q1_target'])

        return out


    def act(self, state: Tensor) -> NDArray:
        # choose action
        q = self.q1.get_q(state, self.actions_onehot)

        i_action = q.argmax(-1).cpu().numpy()

        return i_action


    def mlflow_get_sample_weights(self) -> dict[str, float]:
        out = {
            "weight_q1": float(self.q1.layers[0].weight[0, 0]), # type: ignore
            "weight_q1_target": float(self.q1_target.layers[0].weight[0, 0]), # type: ignore
        }

        return out


    @property
    def active_submodels(self) -> list[nn.Module]:
        return [self.q1]


    @property
    def target_submodels(self) -> list[nn.Module]:
        return [self.q1_target]


    def training_session(
        self,
        replay_buffer: ReplayBuffer,
        n_epochs: int,
        batch_size: int,
        lr: float,
        gamma: float,
    ):
        mean_loss_q1 = 0
        mean_q = 0

        self.optim_q1.param_groups[0]['lr'] = lr

        for _ in range(n_epochs):
            # sample from replay buffer
            to_tensor = lambda x: tc.stack(x)
            state, action, reward, next_state, terminated, truncated = map(to_tensor, replay_buffer.sample(batch_size))

            # compute target q-value for q-networks
            with tc.no_grad():
                # gauge q-values
                q_target = self.q1_target.get_max_q(next_state[~terminated], self.actions_onehot)

                # compute bellman function
                y = self.bellman_fn(reward, q_target, gamma, terminated)

            # perform gradient step for q1-network
            q1 = self.q1(state, action)
            loss_q1 = self.loss_fn(y.detach(), q1)
            self.optim_q1.zero_grad()
            loss_q1.backward()
            self.optim_q1.step()

            # track losses
            mean_loss_q1 += float(loss_q1)
            mean_q += float(q_target.mean())

        mean_loss_q1 /= n_epochs
        mean_q /= n_epochs

        train_metrics = {
            'loss_q1': mean_loss_q1,
            'q': mean_q
        }

        return train_metrics
