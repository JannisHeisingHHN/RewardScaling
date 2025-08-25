import torch as tc
from torch import nn

from ..ffnn import FFNN

from typing import Iterable, SupportsFloat
from torch.types import Device


class GaussianPolicy(nn.Module):
    def __init__(self, architecture: list[int | nn.Module], out_size: int, lower_bounds: Iterable[SupportsFloat], upper_bounds: Iterable[SupportsFloat], device: Device, *args, **kwargs) -> None:
        '''
        * lower_bounds: lower bounds of the action space, one entry per dimension
        * upper_bounds: upper bounds of the action space, one entry per dimension
        '''
        super().__init__(*args, **kwargs)

        self.ffnn = FFNN(architecture).to(device)
        last_size = [a for a in architecture if isinstance(a, int)][-1]

        self.out_size = int(out_size)
        self.mu = nn.Sequential(self.ffnn.nonlinearity(), nn.Linear(last_size, out_size)).to(device)
        self.log_sigma = nn.Sequential(self.ffnn.nonlinearity(), nn.Linear(last_size, out_size)).to(device)

        self.action_shift = tc.tensor(lower_bounds, device=device)
        self.action_scale = 0.5 * (tc.tensor(upper_bounds, device=device) - tc.tensor(lower_bounds, device=device))

        self.device = device
    

    def forward(self, X: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
        '''X must be a batch'''
        b = X.shape[0]

        # determine mean and standard deviation
        Z = self.ffnn(X)
        mu = self.mu(Z)
        log_sigma = self.log_sigma(Z)
        sigma = tc.exp(log_sigma).clamp(min=1e-2, max=2) # we take the exponential to make sure that the value is positive

        # sample actions. Notation comparisons in parentheses are w.r.t. Appendix C of the SAC paper
        pre_actions = tc.randn(self.out_size, device=self.device) * sigma + mu # unbounded sample (u)
        actions = self.action_scale * (tc.tanh(pre_actions) + 1) + self.action_shift # bounded sample (a)

        # compute log likelihood. The last summand doesn't contribute to the gradient but is included so that the values are interpretable
        ll_preactions = -(pre_actions - mu)**2 / (2 * sigma**2) - tc.log(sigma) - 0.5 * tc.log(2 * tc.pi * tc.ones_like(mu)) # log likelihood of gaussian sample (mu(u | s))
        ll_cov = -tc.log(1 - tc.tanh(pre_actions)**2 + 1e-6) # change-of-variables contribution (da/du)
        ll_scaling = -tc.log(self.action_scale).unsqueeze(0).repeat((b, 1)) # action scaling contribution (not in the paper and doesn't contribute to gradient. purely for testing)

        # # definitely correct implementation
        # dist = tc.distributions.Normal(mu, sigma)
        # pre_actions = dist.rsample() # unbounded sample (u)
        # # pre_actions = tc.randn((b, self.out_size), device=self.device) * sigma + mu # unbounded sample (u)
        # actions = self.action_scale * (tc.tanh(pre_actions) + 1) + self.action_shift # bounded sample (a)

        # ll_preactions = dist.log_prob(pre_actions)
        # ll_cov = -tc.log(1 - tc.tanh(pre_actions)**2)
        # ll_scaling = -tc.log(self.action_scale).unsqueeze(0).repeat((b, 1)) # action scaling contribution (not in the paper and doesn't contribute to gradient. purely for testing so the values are correct)

        # accumulate likelihoods of different action entries for each action in the batch
        # since these are log-likelihoods, they are summed instead of multiplied
        # log_likelihood = (ll_preactions + ll_cov).sum(-1)
        log_likelihood = (ll_preactions + ll_cov + ll_scaling).sum(-1)

        return actions, log_likelihood
