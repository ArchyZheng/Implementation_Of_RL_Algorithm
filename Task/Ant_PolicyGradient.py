import gymnasium as gym
import numpy as np
import torch

from RL_Policy.PolicyGradient import PolicyGradient


def main():
    env = gym.make('CartPole-v1', render_mode='human')
    BATCH_SIZE = 100
    current_times_of_task = 0
    memory = [{} for _ in range(BATCH_SIZE)]
    policy_gradient = PolicyGradient(env, memory)
    for _ in range(10000):
        observation, info = env.reset()
        ob_batch = []
        action_batch = []
        rewards_batch = []
        while True:
            action = policy_gradient(observation).sample().detach().numpy()
            action_batch.append(action)
            next_observation, reward, done, truncated, info = env.step(action)
            rewards_batch.append(reward)
            ob_batch.append(observation)
            observation = next_observation
            if done or truncated:
                current_index = current_times_of_task % BATCH_SIZE
                memory[current_index]['observation'] = ob_batch
                memory[current_index]['action'] = action_batch
                memory[current_index]['return'] = sum(rewards_batch)
                current_times_of_task += 1
                if current_times_of_task % BATCH_SIZE == 0:
                    policy_gradient.train()
                break


if __name__ == '__main__':
    main()
