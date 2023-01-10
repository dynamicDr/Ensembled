import math

import copy
import gym
import numpy as np
import torch

from ddpg import DDPG
from rsoccer_gym.ssl import *

number = 20
step_k = 1801
max_episode = 10
display = True

args = np.load(f"./models/args/args_num_{number}.npy",allow_pickle=True)
agent = DDPG(args)
agent.actor.load_state_dict(torch.load(f"./models/{args.env_name}/{number}/{args.env_name}_{number}_{step_k}k_actor.npy"))
agent.eval()

env = gym.make('SSLShootEnv-v0')


for episode in range(max_episode):
    obs = env.reset()
    terminate = False
    done = False
    episode_step = 0
    episode_reward = 0
    while not (done or terminate):

        # For each step...
        action = agent.select_action(obs)
        obs_next, reward, done, info = env.step(copy.deepcopy(action))
        print(action)
        print(reward)
        if display:
            env.render()
        obs = obs_next

        episode_step += 1
        episode_reward += reward

        if episode_step >= args.episode_limit:
            terminate = True

    episode += 1
    avg_reward = episode_reward / episode_step
    print(f"============epi={episode},avg_reward={avg_reward}==============")