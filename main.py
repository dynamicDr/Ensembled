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
        # action = env.action_space.sample()
        action= np.array([0.5,0.5,0.5,0,1])
        next_state, reward, done, _ = env.step(action)
        # env.render()
        # print("ball", next_state[0], next_state[1])
        # for i in range(3):
        #     print("robot blue",i,next_state[4 + i * 7],next_state[5 + i * 7])
        #     print("robot yellow",i,next_state[24+i*2],next_state[25+i*2])

    print(reward)

