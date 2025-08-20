import torch as tc
from torch import nn

from .ffnn import FFNN
from .learner import Learner
from .replay_buffer import ReplayBuffer

from typing import Callable
from numpy.typing import NDArray
from torch.types import Device
from torch import Tensor


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
        self.loss_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor] = nn.SmoothL1Loss()

        self.q1_target = self.q1.clone()

        self.n_actions = n_actions
        self.actions_onehot = tc.eye(n_actions)

        # The following if-else morally belongs in training_session (where self.bellman_fn is used) as its purpose is to determine
        # how to calculate the Q-functions' targets, but because I'm obsessed with runtime optimization I moved it here
        self.bellman_fn = _bellmann_scaled if use_reward_scaling else _bellmann_std
        self.use_reward_scaling = use_reward_scaling

        self.set_device(device)


    def set_device(self, device: Device):
        self.to(device)
        self.actions_onehot = self.actions_onehot.to(device)
        self.device = device

        return self


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


    def act(self, state: Tensor, actions: Tensor) -> NDArray:
        # choose action
        q = self.q1.get_q(state, actions)

        i_action = q.argmax(-1).cpu().numpy()

        return i_action


    # def target_update_polyak(self, rho):
    #     for q, q_target in zip([self.q1], [self.q1_target]):
    #         v = tc.nn.utils.parameters_to_vector(q.parameters())
    #         v_target = tc.nn.utils.parameters_to_vector(q_target.parameters())

    #         v_new = rho * v_target + (1 - rho) * v

    #         tc.nn.utils.vector_to_parameters(v_new, q_target.parameters())


    # def target_update_copy(self):
    #     for q, q_target in zip([self.q1], [self.q1_target]):
    #         v = tc.nn.utils.parameters_to_vector(q.parameters())
    #         tc.nn.utils.vector_to_parameters(v, q_target.parameters())


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
                # # gauge q-values
                # q_target = self.q1_target.get_max_q(next_state[~terminated], self.actions_onehot)

                # # compute bellman function # TODO revert
                # y = self.bellman_fn(reward, q_target, gamma, terminated)
                q_target = self.q1_target.get_max_q(next_state, self.actions_onehot)
                y = reward + gamma * q_target

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
