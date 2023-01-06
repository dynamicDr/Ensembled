import math

import gym
import numpy as np
from rsoccer_gym.ssl import *

env = gym.make('SSLShootEnv-v0')
env.reset()

for i in range(1):
    done = False
    frame = 0

    while not done:
        # print("Frame",frame)
        frame+=1
        if frame >= 100:
            break
        # action = env.action_space.sample()
        if frame<50:
            action= np.array([0,0,-0.5,0])
        else:
            action = np.array([0, 0,-0.5,1])
        next_state, reward, done, info = env.step(action)
        env.render()
        print(reward)
        # print("frame",frame,"reward",reward)
    print(info)