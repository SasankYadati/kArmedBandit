import numpy as np
from kArmedBandit.agents.utils.utils import argmax

# non associative epsilon greedy agent
class EpsilonGreedyAgent:
    def __init__(self, num_actions=10, epsilon=0.1, step_size_callback=None):
        self.last_action = None
        self.action_values = np.zeros(num_actions)
        self.action_count = np.zeros(num_actions)
        self.epsilon = epsilon
        self.step_size_callback = step_size_callback
    
    def update_action_values(self, action, reward):
        self.action_count[action] += 1
        if self.step_size_callback is None:
            step_size = 1 / self.action_count[action] # default behavior is averaging samples
        else:
            step_size = self.step_size_callback(self.action_count, action)
        self.action_values[action] += step_size*(reward - self.action_values[action])

    def policy(self, reward):        
        should_explore = np.random.random() < self.epsilon
        if should_explore:
            current_action = np.random.randint(0,len(self.action_count))
        else:
            current_action = argmax(self.action_values)
    
        self.update_action_values(current_action, reward)
        return current_action
    
    def reset(self):
        self.last_action = None
        self.action_values *= 0.0
        self.action_count *= 0.0
