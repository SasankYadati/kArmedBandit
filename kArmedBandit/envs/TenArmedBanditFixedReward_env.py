import gym
from gym import spaces
from gym.utils import seeding
import random


class TenArmedBanditFixedRewardEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_stationary, seed=42):
        self._seed(seed)
        self.num_bandits = 10
        self.is_stationary = is_stationary
        self.initialize_rewards()
        self.action_space = spaces.Discrete(self.num_bandits)
        self.observation_space = spaces.Discrete(1)

    def _seed(self, seed=None):
        random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def initialize_rewards(self):
        if self.is_stationary:
            self.rewards = [random.normalvariate(0, 1) for i in range(self.num_bandits)]
        else:
            self.rewards = [random.normalvariate(0, 1)] * self.num_bandits

    def update_reward_dist(self):
        # nudge rewards for each action after each stop making the reward distribution non-stationary
        for i in range(len(self.rewards)):
            shift = random.normalvariate(0, 0.01)
            self.rewards[i] += shift

    def step(self, action):
        assert self.action_space.contains(action)
        reward = random.normalvariate(self.rewards[action], 1)
        done = True
        not self.is_stationary and self.update_reward_dist()
        return 0, reward, done, {}

    def reset(self):
        self.initialize_rewards()
        return 0

    def render(self, mode='human'):
        pass

    def close(self):
        pass
