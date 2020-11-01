import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class TenArmedBanditGaussianRewardEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=42):
        self._seed(seed)
        self.num_bandits = 10
        # each reward distribution is a gaussian described using mean and standard deviation
        self.reward_dist = [[np.random.normal(0, 1), 1] for _ in range(self.num_bandits)]
        self.action_space = spaces.Discrete(self.num_bandits)
        self.observation_space = spaces.Discrete(1)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        reward = 0
        done = True

        # sample reward using the corresponding reward distribution
        reward = np.random.normal(
            self.reward_dist[action][0], self.reward_dist[action][1])

        return 0, reward, done, {}

    def reset(self):
        return 0

    def render(self, mode='human'):
        pass

    def close(self):
        pass
