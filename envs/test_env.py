import gymnasium as gym
import numpy as np
from typing import Any


class TestEnv(gym.Env):
    def __init__(self, target: float | None = None, radius: float | None = None) -> None:
        super().__init__()

        self.target = target
        self.radius = radius

        match target is not None, radius is not None:
            case  True,  True: self.reward_setting = 3
            case  True, False: self.reward_setting = 2
            case False,  True: raise ValueError("If radius is set, target must be set as well!")
            case False, False: self.reward_setting = 1

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space: gym.spaces.Box = gym.spaces.Box(float('-inf'), float('inf'))


    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        self.count = np.zeros(1, dtype=np.float32)

        return self.state, {}


    def step(self, action):
        self.count += action - 1

        match self.reward_setting:
            case 3:
                # target + radius reward
                pre_reward = self.radius - abs(self.count[0] - self.target)
                reward = float(pre_reward if pre_reward > 0 else 0)
            case 2:
                # target reward
                reward = -abs(self.count[0] - self.target)
            case 1:
                # simple reward
                reward = float(action - 1)

        return self.state, reward, False, False, {}


    @property
    def state(self):
        return self.count
