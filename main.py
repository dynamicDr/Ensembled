import math

import gym
import numpy as np
from rsoccer_gym.ssl import *

env = gym.make('SSLShootEnv-v0')


for i in range(10):
    env.reset()
    done = False
    frame = 0
    env.render()
    while not done:
        # print("Frame",frame)
        frame+=1
        if frame >= 50:
            break
        # action = env.action_space.sample()
        action = np.array([1,1,1,0])
        next_state, reward, done, info = env.step(action)
        env.render()
        # print("ball_grad",info["rw_ball_grad"])
        # print("ball_dist",info["rw_ball_dist"])
        # print("frame",frame,"reward",reward)
