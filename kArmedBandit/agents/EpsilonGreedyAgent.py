import numpy as np
from kArmedBandit.agents.utils.utils import argmax
class EpsilonGreedyAgent:
    def __init__(self, num_actions=10, epsilon=0.1):
        self.last_action = None
        # action values are referred to as q-values, initialized with
        self.q_values = num_actions*[0]
        # keep track of action counts
        self.action_count = num_actions*[0]
        # exploration-exploitation tradeoff
        self.epsilon = epsilon
    
    def policy(self, observation, reward):
        if (self.last_action):
            self.action_count[self.last_action] += 1
            self.q_values[self.last_action] += (1/self.action_count[self.last_action])*(reward - self.q_values[self.last_action])
        
        should_explore = np.random.random()
        if should_explore < self.epsilon:
            current_action = np.random.randint(0,len(self.action_count))
        else:
            current_action = argmax(self.q_values)
    
        self.last_action = current_action
        
        return current_action
