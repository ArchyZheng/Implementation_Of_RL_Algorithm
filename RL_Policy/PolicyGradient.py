from collections import namedtuple
import torch
from torch.distributions import Distribution, Normal, Categorical
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size_list=None, hidden_activation=nn.ReLU,
                 output_activation=nn.Identity, action_type='discrete'):
        super(PolicyNetwork, self).__init__()
        if hidden_size_list is None:
            hidden_size_list = [128, 128, 64]
        assert action_type in ['discrete', 'continuous']
        self.log_std = nn.Parameter(torch.zeros(size=(action_size,)))
        self.action_type = action_type
        layers = []
        size_list = [state_size] + hidden_size_list + [action_size]
        for j in range(1, len(size_list)):
            if j < len(size_list) - 1:
                layers += [nn.Linear(size_list[j - 1], size_list[j]), hidden_activation()]
            else:
                layers += [nn.Linear(size_list[j - 1], size_list[j]), output_activation()]

        self.model = nn.Sequential(*layers)

    def forward(self, state) -> Distribution:
        state = torch.from_numpy(state).float()

        if self.action_type == 'discrete':
            return Categorical(logits=self.model(state))
        else:
            return Normal(self.model(state), torch.exp(self.log_std))


class PolicyGradient:
    def __init__(self, env, memory):
        action_dim = env.action_space.n
        observation_dim = env.observation_space.shape[0]

        from torch.optim import Adam
        import itertools

        self.memory = memory
        LR = 1e-3
        self.BATCH_SIZE = 100
        self.policy_network = PolicyNetwork(observation_dim, action_dim)
        if self.policy_network.action_type == 'continuous':
            self.optimizer = Adam(
                params=(itertools.chain([self.policy_network.log_std], self.policy_network.parameters())), lr=LR)
        else:
            self.optimizer = Adam(params=self.policy_network.parameters(), lr=LR)
        self.summary = SummaryWriter()
        self.global_step = 0

    def train(self):
        loss = 0
        number_action = 0
        for i in range(self.BATCH_SIZE):
            state_batch = np.array(self.memory[i]['observation'])
            action_batch = torch.as_tensor(np.array(self.memory[i]['action']))
            number_action += action_batch.shape[0]
            loss = -1 * self.policy_network(state_batch).log_prob(action_batch).sum() * self.memory[i]['return'] + loss
        loss = loss / number_action
        self.summary.add_scalar("loss", loss, self.global_step)
        self.global_step += 1
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def __call__(self, state):
        return self.policy_network(state)
