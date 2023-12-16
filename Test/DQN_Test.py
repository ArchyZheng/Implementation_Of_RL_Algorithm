import unittest
from RL_Policy.DQN import ReplayMemory
from RL_Policy.DQN import DQN, Transition
import torch


class TestReplayMemory(unittest.TestCase):
    def test_append_some(self):
        memory = ReplayMemory(10)
        state = 10
        action = 12
        next_state = 14
        reward = 16
        memory.push(state, action, next_state, reward)

    def test_sample_some(self):
        state = torch.rand(size=(512, 4))
        action = torch.randint(low=0, high=2, size=(512, 1))
        next_state = torch.rand(size=(512, 4))
        reward = torch.rand(size=(512, 1))
        # array = torch.cat([state, action, next_state, reward], dim=1)
        memory = ReplayMemory(capacity=256)
        dqn = DQN(memory=memory, batch_side=128)
        for index in range(512):
            memory.push(state[index], action[index], next_state[index], reward[index])
            dqn.optimal_model()
            aciton = dqn.optimal_action(state)


if __name__ == '__main__':
    unittest.main()
