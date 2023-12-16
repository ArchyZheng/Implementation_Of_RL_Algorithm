# %%
import gymnasium as gym
from RL_Policy.DQN import DQN, ReplayMemory
import random
import torch
from torch.utils.tensorboard import SummaryWriter
import math

if __name__ == '__main__':

    env = gym.make("CartPole-v1")
    observation, info = env.reset()
    replay_memory = ReplayMemory(capacity=10000)
    dqn = DQN(memory=replay_memory, batch_side=128)
    summary_writer = SummaryWriter()
    episode_number = 0
    EPS_END = 0.05
    EPS_START = 0.9
    EPS_DECAY = 1000

    steps_done = 0


    def select_action(observation):
        global steps_done
        random_number = random.random()
        epsilon_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if random_number > epsilon_threshold:
            action = dqn.optimal_action(observation).item()
        else:
            action = env.action_space.sample()
        return action


    reward_list = []

    for _ in range(1000000):
        # env.render()
        action = select_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)
        reward_list.append(reward)

        replay_memory.push(torch.tensor(observation), torch.tensor([action]), torch.tensor(next_observation),
                           torch.tensor([reward]))

        observation = next_observation
        if terminated or truncated:
            summary_writer.add_scalar("Reward", torch.tensor(reward_list).sum(), episode_number)
            episode_number += 1
            reward_list = []
            observation, info = env.reset()

        random_number_for_update = random.random()
        # if random_number_for_update < 0.5:
        dqn.optimal_model()
    env.close()
