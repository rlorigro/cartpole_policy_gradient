from handlers.FileManager import FileManager
from models.RNNPolicyGradient3 import Policy
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import gym
import sys
import os


def get_timestamp_string():
    time = datetime.now()
    year_to_second = [time.year, time.month, time.day, time.hour, time.second]

    time_string = "-".join(map(str,year_to_second))

    return time_string


class ModelHandler:
    def __init__(self, output_directory):
        FileManager.ensure_directory_exists(output_directory)

        self.output_directory = output_directory
        self.time_string = get_timestamp_string()
        self.filename = "model_" + self.time_string

    def save_model(self, model):
        path = os.path.join(self.output_directory, self.filename)

        print("SAVING MODEL:", path)
        torch.save(model.state_dict(), path)


def plot_results(policy):
    n_episodes = len(policy.reward_history)
    window = int(n_episodes / 20)

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharey=True, figsize=[9, 9])
    rolling_mean = pd.Series(policy.reward_history).rolling(window).mean()
    std = pd.Series(policy.reward_history).rolling(window).std()
    ax1.plot(rolling_mean)
    ax1.fill_between(range(len(policy.reward_history)), rolling_mean - std, rolling_mean + std, color='orange', alpha=0.2)
    ax1.set_title('Episode Length Moving Average ({}-episode window)'.format(window))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Length')

    ax2.plot(policy.reward_history)
    ax2.set_title('Episode Length')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')

    fig.tight_layout(pad=2)
    plt.show()
    plt.close()

    # fig.savefig('results.png')


def train(n_episodes, policy, environment, optimizer, model_handler, render=False):
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

            # Save reward
            policy.reward_episode.append(reward)

            if done:
                break

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (time * 0.01)

        update_policy(policy=policy, optimizer=optimizer, e=e)

        if e % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(e, time, running_reward))

        if e % 250 == 0:
            model_handler.save_model(model=policy)

        if running_reward > environment.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,time))
            break


def update_policy(policy, optimizer, e):
    R = 0
    rewards = []

    # Discount future rewards back to the present using gamma
    for r in reversed(policy.reward_episode):
        R = r + policy.gamma * R
        rewards.insert(0, R)

    # Scale rewards
    rewards = torch.FloatTensor(rewards)

    # print(rewards)

    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))

    # print(len(policy.policy_history))

    # Calculate loss
    policy_history = torch.stack(policy.policy_history)
    loss = torch.mul(policy_history, rewards).mul(-1)

    # print(loss)
    loss = torch.sum(loss, dim=-1)
    # print(loss)

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and intialize episode history counters
    policy.loss_history.append(loss.data[0])
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.reset_episode()


def select_action(policy, state):
    # print("ACTION SELECTION")

    # print("state:", state.shape)

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
    model_handler = ModelHandler(output_directory=OUTPUT_DIRECTORY)

    render = False

    # gym.envs.register(
    #     id='CartPole-v666',
    #     entry_point='gym.envs.classic_control:CartPoleEnv',
    #     max_episode_steps=5000,
    #     tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
    #     reward_threshold=5000.0,
    # )

    # Initialize environment
    environment = gym.make('CartPole-v1')
    # environment = gym.make('CartPole-v666')
    # environment._max_reward_episode = 5000
    # environment.reward_threshold = 5000

    environment.seed(1)
    torch.manual_seed(1)

    # Access environment/agent variables
    state_space = environment.observation_space
    action_space = environment.action_space

    # Hyperparameters
    n_episodes = sys.maxsize
    learning_rate = 1e-3
    weight_decay = 1e-4
    gamma = 0.99

    # Architecture parameters
    n_layers = 3
    hidden_size = 128

    policy = Policy(action_space=action_space,
                    state_space=state_space,
                    hidden_size=hidden_size,
                    n_layers=n_layers,
                    dropout_rate=0.6,
                    gamma=gamma)
    print(policy)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train(n_episodes=n_episodes, environment=environment, policy=policy, optimizer=optimizer, model_handler=model_handler, render=render)

    plot_results(policy=policy)

    model_handler.save_model(model=policy)


if __name__ == "__main__":
    OUTPUT_DIRECTORY = "/home/ryan/code/RL/output/"

    run()
