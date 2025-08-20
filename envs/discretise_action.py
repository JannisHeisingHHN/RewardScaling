import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from functools import cache


# copied and slightly altered from a file Simon sent me
class DiscretiseAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env, n_actions: int):
        super().__init__(env)

        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("Environment must have continuous action space.")
        if n_actions < 2:
            raise ValueError("Need at least two actions for discretisation.")

        self.action_space = gym.spaces.Discrete(n_actions)
        high = env.action_space.high
        low = env.action_space.low
        self.c = low
        self.m = (high - low) / (n_actions - 1)

    @cache
    def action(self, action: int | NDArray[np.int32]) -> float |NDArray[np.float64]:
        return self.m * action + self.c
