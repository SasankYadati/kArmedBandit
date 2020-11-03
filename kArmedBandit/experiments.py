import numpy as np
import matplotlib.pyplot as plt
import gym
import kArmedBandit
from kArmedBandit.agents.EpsilonGreedyAgent import EpsilonGreedyAgent
from kArmedBandit.agents.EpsilonGreedyUCBAgent import EpsilonGreedyUCBAgent

num_runs = 200
num_steps = 1000

epsilons = [0, 0.2, 0.5, 0.8, 1]

epsilon_greedy_agents = [EpsilonGreedyAgent(
    10, epsilon) for epsilon in epsilons]
epsilon_greedy_ucb_agents = [EpsilonGreedyUCBAgent(
    10, epsilon) for epsilon in epsilons]

TEN_ARMED_FIXED = "TenArmedBanditFixed-v0"
TEN_ARMED_GAUSSIAN = "TenArmedBanditGaussian-v0"

ten_armed_fixed_env = {
    "name": 'TenArmedBanditFixed-v0',
    "env": gym.make("TenArmedBanditFixed-v0"),
    "all_averages": {
        "eps_greedy_agent": []
    }
}

ten_armed_gaussian_env = {
    "name": 'TenArmedBanditFixed-v0',
    "env": gym.make("TenArmedBanditFixed-v0"),
    "all_averages": {
        "eps_greedy_agent": [],
        "eps_greedy_ucb_agent": []
    }
}


def run_experiment(env, agent):
    for run in range(num_runs):
        np.random.seed(run)
        score = 0
        averages = []
        for i in range(num_steps):
            action = agent.policy(0,0)
            obs, reward, _, _ = env.step(action)
            score += reward
            averages.append(score / (i + 1))
    return averages


def plot():
    plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.plot([1.55 for _ in range(num_steps)], linestyle="--")
    plt.title("Averages of Epsilon-Greedy Agent")
    plt.plot(np.mean(all_averages, axis=0))
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.show()


# experiment on ten_armed_fixed_env
for epsilon in epsilons:
    avg = run_experiment(
        ten_armed_fixed_env["env"], EpsilonGreedyAgent(10, epsilon))
    ten_armed_fixed_env["all_averages"]["eps_greedy_agent"].append(avg)

# experiment on ten_armed_gaussian_env
for epsilon in epsilons:
    avg = run_experiment(
        ten_armed_gaussian_env["env"], EpsilonGreedyAgent(10, epsilon))
    ten_armed_gaussian_env["all_averages"]["eps_greedy_agent"].append(avg)
    avg = run_experiment(
        ten_armed_gaussian_env["env"], EpsilonGreedyUCBAgent(10, epsilon))
    ten_armed_gaussian_env["all_averages"]["eps_greedy_ucb_agent"].append(avg)
