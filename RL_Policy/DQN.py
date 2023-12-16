import torch
from collections import namedtuple, deque
import torch.nn as nn
from torch.optim import Adam
import random

Transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward'))


class ActionValueNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layer_sizes: list):
        super(ActionValueNetwork, self).__init__()
        self.input_size = input_size
        layers = []
        sizes = [input_size, *hidden_layer_sizes, output_size]
        act_hidden = nn.Tanh
        act_out = nn.Identity
        for j in range(len(sizes) - 1):
            act = act_hidden if j < len(sizes) - 2 else act_out
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        self.action_value_network = nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        assert input_tensor.shape[-1] == self.input_size
        return self.action_value_network(torch.tensor(input_tensor))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        import random
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:

    def __init__(self, memory: ReplayMemory, batch_side: int):
        """

        :param memory:
        :param batch_side:
        """
        self._memory = memory
        self._batch_size = batch_side

        self.action_value_network = ActionValueNetwork(input_size=4, output_size=2, hidden_layer_sizes=[128, 128])
        self.target_network = ActionValueNetwork(input_size=4, output_size=2, hidden_layer_sizes=[128, 128])
        self.target_network.load_state_dict(self.action_value_network.state_dict())
        self.gamma = 0.99
        self.optimizer = Adam(params=self.action_value_network.parameters(), lr=1e-3)

    def optimal_action(self, input_state):
        with torch.no_grad():
            return self.action_value_network(torch.tensor(input_state).view(-1, 4)).max(1)[1]

    def optimal_model(self):
        if len(self._memory) < self._batch_size:
            return

        transitions = self._memory.sample(self._batch_size)
        # How to understand this?
        batch: Transition = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])
        state_batch: torch.Tensor = torch.cat(batch.state)
        action_batch: torch.Tensor = torch.cat(batch.action)
        reward_batch: torch.Tensor = torch.cat(batch.reward)

        action_value = self.action_value_network(state_batch.view(-1, 4)).gather(1, action_batch.view(-1, 1))
        next_state_value = torch.zeros_like(action_value)
        with torch.no_grad():
            next_state_value[non_final_mask] = \
            self.target_network(non_final_next_state.view(-1, 4)).max(dim=1, keepdim=True)[0]

        expected_state_value = next_state_value * self.gamma + reward_batch.view(-1, 1)[non_final_mask]

        criterion = nn.SmoothL1Loss()
        loss = criterion(action_value, expected_state_value)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.action_value_network.parameters(), 100)
        self.optimizer.step()
        # soft update
        target_state_dict = self.target_network.state_dict()
        action_sate_dict = self.action_value_network.state_dict()
        tau = 0.005
        for key in action_sate_dict:
            target_state_dict[key] = tau * action_sate_dict[key] + (1 - tau) * target_state_dict[key]
        self.target_network.load_state_dict(target_state_dict)
