import math

import copy
import gym
import numpy as np
import torch

from ddpg import DDPG
from rsoccer_gym.ssl import *


number = 39
step_k = 2347
max_episode = 1000
display = False

print(f"match for number{number} {step_k}k ckp")

args = np.load(f"./models/args/args_num_{number}.npy",allow_pickle=True)
agent = DDPG(args)
agent.actor.load_state_dict(torch.load(f"./models/{args.env_name}/{number}/{args.env_name}_{number}_{step_k}k_actor.npy"))
agent.eval()

env = gym.make('SSLShootEnv-v0')

goal_num = 0
fault_goal_num = 0
done_rbt_out = 0
done_ball_out = 0
done_time_up = 0
avg_episode_step = 0
for episode in range(max_episode):
    obs = env.reset()
    terminate = False
    done = False
    episode_step = 0
    episode_reward = 0
    while not (done or terminate):

        # For each step...
        action = agent.select_action(obs,0)
        obs_next, reward, done, info = env.step(copy.deepcopy(action))
        # print(action)
        # print(reward)
        if display:
            env.render()
        obs = obs_next

        episode_step += 1
        episode_reward += reward

        if episode_step >= args.episode_limit:
            terminate = True

    episode += 1

    avg_reward = episode_reward / episode_step
    # print(f"============epi={episode},avg_reward={avg_reward}==============")
    if info["goal"] == 1:
        goal_num+=1
    elif info["goal"] == -1:
        fault_goal_num +=1
    if info["done_robot_out"] == 1:
        done_rbt_out+=1
    elif info["done_ball_out"] == 1:
        done_ball_out +=1
    avg_episode_step +=episode_step
avg_episode_step /= max_episode
print("goal",goal_num,"opp_goal",fault_goal_num)
print("done_ball_out",done_ball_out,"done_rbt_out",done_rbt_out,"done_other",max_episode-(goal_num+fault_goal_num+done_ball_out+done_rbt_out))
print("avg_episode_step",avg_episode_step)