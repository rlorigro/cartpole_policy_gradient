from models.PolicyGradient import Policy
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import math
import gym
import sys
import os

'''
This is a rewrite of the script originally written by git user ts1839, and documented here:
https://medium.com/@ts1829/policy-gradient-reinforcement-learning-in-pytorch-df1383ea0baf
'''


def plot_results(n_episodes, policy):
    window = int(n_episodes / 20)

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
    rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
    std = pd.Series(policy.reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(policy.reward_history)), rolling_mean - std, rolling_mean + std, color='orange',
                     alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')

    ax2.plot(policy.reward_history)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    plt.show()

    # fig.savefig('results.png')


def test(n_episodes, policy, environment, render=False):
    running_reward = 10

    for e in range(n_episodes):
        state = environment.reset()  # Reset environment and record the starting state

        time = 0
        done = False

        # run a simulation for up to 1000 time steps
        for time in range(1000):
            if render:
                environment.render()

            # print("TIME: ", time)
            action = select_action(policy=policy, state=state)

            # Step through environment using chosen action, "done" means simulation reached termination condition
            state, reward, done, _ = environment.step(action)

            if time % 1 == 0:
                sys.stdout.write("\r%d"%time)
            # print(time)
            # print('\t'.join(map(lambda x: str(round(x,3)),state)))

            # sys.stdin.readline()

        if e % 50 == 0:
            # plt.show()
            # plt.close()
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(e, time, running_reward))


def select_action(policy, state):
    # print("ACTION SELECTION")

    # perform forward calculation on the environment state variables to obtain the probability of each action state
    state = torch.FloatTensor(state)
    action_state = policy.forward(state)

    # Sample based on the output probabilities of the model
    choice_distribution = Categorical(action_state)
    action = choice_distribution.sample()

    # Add log probability of our chosen action to history
    action_probability = choice_distribution.log_prob(action)
    policy.policy_history.append(action_probability)

    return action.item()


def run():
    render = True

    model_path = "output/model_2018-12-15-16-56"
    model_path = os.path.abspath(model_path)

    print("LOADING MODEL: ", model_path)

    # Initialize environment
    environment = gym.make('CartPole-v1')
    environment.seed(1)
    torch.manual_seed(1)

    # Access environment/agent variables
    state_space = environment.observation_space
    action_space = environment.action_space

    # Hyperparameters
    n_episodes = 1500
    gamma = 0.99

    policy = Policy(action_space=action_space, state_space=state_space, dropout_rate=0.6, gamma=gamma)
    policy.load_state_dict(torch.load(model_path))

    print(policy)

    test(n_episodes=n_episodes, environment=environment, policy=policy, render=render)

    plot_results(n_episodes=n_episodes, policy=policy)


if __name__ == "__main__":
    run()
