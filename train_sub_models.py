import gym
import numpy

env = gym.make('SSL3v3Env-v0')
env.reset()

for i in range(1):
    done = False
    frame = 0
    while not done:
        print("Frame",frame)
        frame+=1
        # action = env.action_space.sample()
        # print(type(action))
        action= numpy.zeros(5)
        next_state, reward, done, _ = env.step(action)
        # env.render()
        for i in range(3):
            print("robot b",i,next_state[4 + i * 7],next_state[5 + i * 7])
            print("robot y",i,next_state[24+i*2],next_state[25+i*2])
        print(next_state[40],next_state[41])

    print(reward)

