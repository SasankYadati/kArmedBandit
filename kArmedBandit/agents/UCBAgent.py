import numpy as np
from kArmedBandit.agents.utils.utils import argmax
from math import log

class UCBAgent:
    def __init__(self, num_actions=10, c=2):
        self.last_action = None
        # action values are referred to as q-values, initialized with zeros
        self.q_values = np.zeros(num_actions)
        # keep track of action counts
        self.action_count = np.zeros(num_actions)
        # confidence level
        self.confidence_level = c
    
    def policy(self, observation, reward):
        if (self.last_action is not None):
            self.action_count[self.last_action] += 1
            self.q_values[self.last_action] += (1/self.action_count[self.last_action])*(reward - self.q_values[self.last_action])

        num_steps = np.sum(self.action_count)
        untried_action = self.find_untried_action()
        if untried_action is not None:
            current_action = untried_action
        else:
            uncertainty_est = log(num_steps) / self.action_count
            adj = self.confidence_level * np.sqrt(uncertainty_est)
            q_values_adj = self.q_values + adj
            current_action = argmax(q_values_adj)
    
        self.last_action = current_action
        
        return current_action
    
    def find_untried_action(self):
        untried_actions = np.where(self.action_count == 0)[0]
        if len(untried_actions) == 0:
            return None
        return np.random.choice(untried_actions)

    def reset(self):
        self.last_action = None
        self.q_values *= 0.0
        self.action_count *= 0.0

if __name__ == '__main__':
    agent = UCBAgent(3, 2)
    rewards = [10, -3, 5]
    obs, rew = 0, 0
    for i in range(10):
        action = agent.policy(obs, rew)
        rew = rewards[action]
        print("Action", action,"Value", agent.q_values[action])
    