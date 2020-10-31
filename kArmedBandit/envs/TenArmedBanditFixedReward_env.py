import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random


class TenArmedBanditFixedRewardEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=42):
        self._seed(seed)
        self.num_bandits = 10
        self.reward_dist = []
        for _ in range(self.num_bandits):
            # each action has a fixed reward
            self.reward_dist.append(random.uniform(0, 1))
        self.action_space = spaces.Discrete(self.num_bandits)
        self.observation_space = spaces.Discrete(1)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        reward = self.reward_dist[action]
        done = True

        return 0, reward, done, {}

    def reset(self):
        return 0

    def render(self, mode='human'):
        pass

    def close(self):
        pass
