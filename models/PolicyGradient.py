import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


class Policy(nn.Module):
    def __init__(self, state_space, action_space, dropout_rate, gamma):
        super(Policy, self).__init__()
        self.input_size = state_space.shape[0]
        self.ouput_size = action_space.n

        self.linear1 = nn.Linear(self.input_size, 128, bias=False)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(128, self.ouput_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.gamma = gamma

        # Episode policy and reward history
        self.policy_history = None
        self.reward_episode = None
        self.reward_episode_local = None

        self.reset_episode()

        # Overall reward and loss history
        self.reward_history = list()
        self.reward_history_local = list()
        self.loss_history = list()

    def reset_episode(self):
        # Episode policy and reward history
        self.policy_history = list()
        self.reward_episode = list()
        self.reward_episode_local = list()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)

        return x
