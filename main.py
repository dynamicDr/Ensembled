import math

import gym
import numpy as np
from rsoccer_gym.ssl import *

env = gym.make('SSLShootEnv-v0')

ball_grad_sum = 0

for i in range(1):
    env.reset()
    done = False
    frame = 0
    # input()
    return_ = 0
    while not done:
        # print("Frame",frame)
        frame+=1
        if frame >= 1000:
            break
        # action = env.action_space.sample()
        if frame<10:
            action = np.array([0,0,1,0])
        else:
            action = np.array([0,0,1,1])
        next_state, reward, done, info = env.step(action)
        env.render()
        # ball_grad_sum +=info["rw_robot_grad"]
        # ball_grad_sum += info["rw_energy"]
        print(reward)
        # print("ball_dist",info["rw_ball_dist"])
        # print("frame",frame,"reward",reward)
