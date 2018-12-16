import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, state_space, action_space, hidden_size, n_layers, dropout_rate, gamma):
        super(Policy, self).__init__()
        self.input_size = state_space.shape[0]
        self.output_size = action_space.n
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        print("actions:", self.output_size)

        self.rnn = nn.GRUCell(input_size=self.input_size,
                              hidden_size=self.hidden_size)

        self.relu = nn.LeakyReLU()
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=-1)

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.gamma = gamma

        # history
        self.hidden_history = None
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
        self.hidden_history = list()
        self.policy_history = list()
        self.reward_episode = list()
        self.reward_episode_local = list()

    def forward(self, x):
        # print("input:", x.shape)

        size = x.shape[0]
        x = x.view([1, size])   # batch size = 1

        if len(self.hidden_history) > 0:
            h_0 = self.hidden_history[-1]
        else:
            h_0 = None

        x = self.rnn(x, h_0)
        self.hidden_history.append(x)

        x = x.reshape([self.hidden_size])
        # print(x.shape)

        x = self.relu(x)
        x = self.linear(x)

        # print(x.shape)

        x = self.softmax(x)

        # print("output:", x.shape)

        return x

