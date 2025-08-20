import torch as tc
from torch import Tensor

from .ffnn import FFNN


class QFFNN(FFNN):
    '''Copy of FFNN whose forward function accepts two inputs instead of one and whose output is squeezed (nicer interface for Q-learning)'''
    def forward(self, state: Tensor, action: Tensor):
        X = tc.concat([state, action], dim=-1)
        return super().forward(X).squeeze()
    

    def get_q(self, states: Tensor, actions: Tensor):
        '''
        states has shape (B, D) and actions has shape (M, C), where<br>
        N: batch size<br>
        D: number of features in a state<br>
        M: number of actions (must be constant across states)<br>
        C: number of features in an action<br>
        If `states` is one-dimensional, `N` is assumed to be 1, and likewise for `actions` and `C`
        '''
        # add dimensions if necessary
        states = tc.atleast_2d(states) # batch dimension
        actions = actions.view(len(actions), -1) # action feature dimension

        N = len(states)
        M = len(actions)

        # copy states and actions to match with one-another
        S = states.repeat_interleave(M, 0)
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
