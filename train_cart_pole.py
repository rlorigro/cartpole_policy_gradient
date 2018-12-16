from handlers.FileManager import FileManager
from models.PolicyGradient import Policy
from torch.distributions import Categorical
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import gym
import os


def get_timestamp_string():
    time = datetime.now()
    year_to_second = [time.year, time.month, time.day, time.hour, time.second]

    time_string = "-".join(map(str,year_to_second))

    return time_string


def save_model(output_directory, model):
    FileManager.ensure_directory_exists(output_directory)

    timestamp = get_timestamp_string()
    filename = "model_" + timestamp
    path = os.path.join(output_directory, filename)

    print("SAVING MODEL:", path)
    torch.save(model.state_dict(), path)


def plot_results(n_episodes, policy):
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

    # fig.savefig('results.png')


def train(output_directory, n_episodes, policy, environment, optimizer, render=False):
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
            # plt.show()
            # plt.close()
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(e, time, running_reward))

            save_model(output_directory=output_directory, model=policy)

        if running_reward > environment.spec.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward,
                                                                                                        time))
            break


def update_policy(policy, optimizer, e):
    R = 0
    rewards = []
    reward_sum = 0

    # Discount future rewards back to the present using gamma
    for r in reversed(policy.reward_episode):
        R = r + policy.gamma * R
        rewards.insert(0, R)
        reward_sum += R

    # Scale rewards
    rewards = torch.FloatTensor(rewards)

    # print(rewards)

    # Normalize rewards
    rewards = (rewards - rewards.mean()) / (rewards.std() + float(np.finfo(np.float32).eps))

    # print(len(policy.policy_history))

    # Calculate loss
    policy_history = torch.stack(policy.policy_history)

    # print(torch.mul(policy_history, rewards).mul(-1))

    loss = torch.sum(torch.mul(policy_history, rewards).mul(-1), dim=-1)

    # print(loss)

    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save and initialize episode history counters
    policy.loss_history.append(loss.data.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.reset_episode()

    # if e % 50 == 0:
    #     plt.plot(rewards.data.numpy())
    #     # print(loss.data.numpy())
    #     plt.show()
    #     plt.close()


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
    render = False

    output_directory = "/home/ryan/code/RL/output/"

    gym.envs.register(
        id='CartPole-v666',
        entry_point='gym.envs.classic_control:CartPoleEnv',
        max_episode_steps=5000,
        tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
        reward_threshold=5000.0,
    )

    # Initialize environment
    environment = gym.make('CartPole-v666')
    environment.seed(1)
    torch.manual_seed(1)

    # Access environment/agent variables
    state_space = environment.observation_space
    action_space = environment.action_space

    # Hyperparameters
    n_episodes = 1500
    learning_rate = 1e-2
    weight_decay = 1e-3
    gamma = 0.99

    policy = Policy(action_space=action_space, state_space=state_space, dropout_rate=0.6, gamma=gamma)
    print(policy)

    optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train(output_directory=output_directory, n_episodes=n_episodes, environment=environment, policy=policy, optimizer=optimizer, render=render)

    plot_results(n_episodes=n_episodes, policy=policy)

    save_model(output_directory=output_directory, model=policy)


if __name__ == "__main__":
    run()
