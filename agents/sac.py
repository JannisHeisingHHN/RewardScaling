import torch as tc
from torch import nn

from .qffnn import QFFNN
from .learner import Learner
from .replay_buffer import ReplayBuffer
from .policies import *

from typing import Callable, Any, Self
from torch.types import Device


class SAC(Learner):
    '''
    Implementation based on the paper "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
    as well as the OpenAI's spinning-up page "https://spinningup.openai.com/en/latest/algorithms/sac.html"
    '''
    def __init__(
        self,
        architecture: list[int | nn.Module],
        policy_class: str,
        policy_args: dict[str, Any],
        temperature: float,
        device: Device,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.q1 = QFFNN(architecture)
        self.q2 = QFFNN(architecture)
        pc: type[nn.Module] = eval(policy_class)
        self.pi = pc(**policy_args)

        self.policy_class = policy_class
        self.policy_args = policy_args

        self.optim_q1 = tc.optim.Adam(self.q1.parameters())
        self.optim_q2 = tc.optim.Adam(self.q2.parameters())
        self.optim_pi = tc.optim.Adam(self.pi.parameters())
        self.loss_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor] = nn.MSELoss()

        self.q1_target = self.q1.clone()
        self.q2_target = self.q2.clone()

        self.temperature = temperature

        # TODO add reward scaling

        self.set_device(device)


    def to_dict(self):
        # put all model data into a single dictionary
        model_dict = {
            'architecture': self.q1.architecture,
            'policy_class': self.policy_class,
            'policy_args': self.policy_args,

            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'pi': self.pi.state_dict(),

            'optim_q1': self.optim_q1.state_dict(),
            'optim_q2': self.optim_q2.state_dict(),
            'optim_pi': self.optim_pi.state_dict(),

            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),

            'temperature': self.temperature,
        }

        return model_dict


    @classmethod
    def from_dict(cls, model_dict: dict[str, Any], device: Device) -> Self:
        out = cls(
            architecture = model_dict['architecture'],
            policy_class = model_dict['policy_class'],
            policy_args = model_dict['policy_args'],
            temperature = model_dict['temperature'],
            device = device,
        )

        out.q1.load_state_dict(model_dict['q1'])
        out.q2.load_state_dict(model_dict['q2'])
        out.pi.load_state_dict(model_dict['pi'])

        out.optim_q1.load_state_dict(model_dict['optim_q1'])
        out.optim_q2.load_state_dict(model_dict['optim_q2'])
        out.optim_pi.load_state_dict(model_dict['optim_pi'])

        out.q1_target.load_state_dict(model_dict['q1_target'])
        out.q2_target.load_state_dict(model_dict['q2_target'])

        return out


    def act(self, state):
        '''Let the policy choose an action'''
        # choose action
        action: tc.Tensor = self.pi(state)[0]

        return action


    def mlflow_get_sample_weights(self):
        out = {
            'weight_q1': float(self.q1.layers[0].weight[0, 0]), # type: ignore
            'weight_q2': float(self.q2.layers[0].weight[0, 0]), # type: ignore
            'weight_q1_target': float(self.q1_target.layers[0].weight[0, 0]), # type: ignore
            'weight_q2_target': float(self.q2_target.layers[0].weight[0, 0]), # type: ignore
        }

        return out


    @property
    def active_submodels(self) -> list[nn.Module]:
        return [self.q1, self.q2]
    

    @property
    def target_submodels(self) -> list[nn.Module]:
        return [self.q1_target, self.q2_target]


    def training_session(
        self,
        replay_buffer: ReplayBuffer,
        n_epochs: int,
        batch_size: int,
        lr: float,
        gamma: float,
    ):
        # heavily inspired by the pseudocode at https://spinningup.openai.com/en/latest/algorithms/sac.html#pseudocode
        mean_loss_q1 = 0
        mean_loss_q2 = 0
        mean_loss_pi = 0
        mean_ll = 0
        mean_ll_next = 0
        mean_q = 0
        mean_q_new = 0

        self.optim_q1.param_groups[0]['lr'] = lr
        self.optim_q2.param_groups[0]['lr'] = lr
        self.optim_pi.param_groups[0]['lr'] = lr

        for _ in range(n_epochs):
            # sample from replay buffer
            state, action, reward, next_state, terminated, truncated = replay_buffer.sample(batch_size)

            # --- Critic Training

            # compute target q-value for q-networks
            with tc.no_grad():
                # sample prospective action and its log likelihood
                new_next_action, next_log_likelihood = self.pi(next_state)

                # gauge q-values
                q1_target = self.q1_target(next_state, new_next_action)
                q2_target = self.q2_target(next_state, new_next_action)

                # take minimum to mitigate runoff
                q_target = tc.min(q1_target, q2_target)

                # compute bellman function
                y: tc.Tensor = reward + gamma * (1 - terminated.int()) * (q_target - self.temperature * next_log_likelihood)

            # perform gradient step for q1-network
            q1 = self.q1(state, action)
            loss_q1 = self.loss_fn(y.detach(), q1)
            assert not (loss_q1.isnan() or loss_q1.isinf())
            self.optim_q1.zero_grad()
            loss_q1.backward()
            self.optim_q1.step()

            # perform gradient step for q2-network
            q2 = self.q2(state, action)
            loss_q2 = self.loss_fn(y.detach(), q2)
            assert not (loss_q2.isnan() or loss_q2.isinf())
            self.optim_q2.zero_grad()
            loss_q2.backward()
            self.optim_q2.step()

            # --- Actor Training

            # sample action and its log likelihood
            new_action, log_likelihood = self.pi(state)

            # compute q-value for newly chosen action
            with tc.no_grad():
                # self.q1.eval()
                # self.q2.eval()
                q1_new = self.q1(state, new_action)
                q2_new = self.q2(state, new_action)
                q_new = tc.min(q1_new, q2_new)

            # perform gradient step for policy network. Note that the q-value is to be maximized and likelihood is to be minimized, hence the sign
            loss_pi: tc.Tensor = -(q_new - self.temperature * log_likelihood).mean()
            assert not (loss_pi.isnan() or loss_pi.isinf())
            self.optim_pi.zero_grad()
            loss_pi.backward()
            nn.utils.clip_grad_norm_(self.pi.parameters(), 10) # from Simon, who has it from Pascal
            self.optim_pi.step()

            # self.q1.train()
            # self.q2.train()

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

        train_metrics = {
            'loss_q1': mean_loss_q1,
            'loss_q2': mean_loss_q2,
            'loss_pi': mean_loss_pi,
            'll': mean_ll,
            'll_next': mean_ll_next,
            'q': mean_q,
            'q_new': mean_q_new,
        }

        return train_metrics
