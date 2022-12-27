import torch
import numpy as np


class ReplayBuffer(object):
    def __init__(self, args, opp=False):
        self.N = args.N  # The number of agents
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.count = 0
        self.current_size = 0
        self.buffer_obs=np.empty((self.buffer_size, args.obs_dim))
        self.buffer_a=np.empty((self.buffer_size, args.action_dim))
        self.buffer_r=np.empty((self.buffer_size, 1))
        self.buffer_obs_next=np.empty((self.buffer_size, args.obs_dim))
        self.buffer_done=np.empty((self.buffer_size, 1))

    def store_transition(self, obs, a, r, obs_next, done):
        self.buffer_obs[self.count] = obs
        self.buffer_a[self.count] = a
        self.buffer_r[self.count] = r
        self.buffer_obs_next[self.count] = obs_next
        self.buffer_done[self.count] = 1 if done else 0
        self.count = (self.count + 1) % self.buffer_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, ):
        index = np.random.choice(self.current_size, size=self.batch_size, replace=False)
        batch_obs=torch.tensor(self.buffer_obs[index], dtype=torch.float)
        batch_a=torch.tensor(self.buffer_a[index], dtype=torch.float)
        batch_r=torch.tensor(self.buffer_r[index], dtype=torch.float)
        batch_obs_next=torch.tensor(self.buffer_obs_next[index], dtype=torch.float)
        batch_done=torch.tensor(self.buffer_done[index], dtype=torch.float)

        return batch_obs, batch_a, batch_r, batch_obs_next, batch_done
