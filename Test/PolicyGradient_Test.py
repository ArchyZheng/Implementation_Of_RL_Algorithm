import unittest
from RL_Policy.PolicyGradient import PolicyNetwork
import numpy as np


class MyTestCase(unittest.TestCase):
    @staticmethod
    def test_policy_network():
        policy_network = PolicyNetwork(state_size=10, action_size=4, action_type='continuous')
        state = np.random.random_sample(10, )
        action_distribution = policy_network(state)
        output = action_distribution.sample()
        print(output)


if __name__ == '__main__':
    unittest.main()
