from collections import namedtuple
import torch
from torch.distributions import Distribution, Normal, Categorical
import torch.nn as nn

transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


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
            return Categorical(self.model(state))
        else:
            return Normal(self.model(state), torch.exp(self.log_std))


class PolicyGradient:
    def __init__(self):
