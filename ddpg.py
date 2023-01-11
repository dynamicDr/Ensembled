import numpy as np
import numpy.random
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from utils import *

criterion = nn.MSELoss()

def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1, hidden2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = torch.tanh(self.fc3(x))

        return a

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1, hidden2):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)


    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        out = self.fc2(torch.cat([out, a], 1))
        out = self.relu(out)
        out = self.fc3(out)
        return out


class DDPG(object):
    def __init__(self, args):
        self.evaluation = False
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim

        self.use_grad_clip = args.use_grad_clip

        # Create Actor and Critic Network
        net_cfg = {
            'hidden1': args.hidden_dim_1,
            'hidden2': args.hidden_dim_2
        }
        self.actor = Actor(self.obs_dim, self.action_dim, **net_cfg)
        self.actor_target = Actor(self.obs_dim, self.action_dim, **net_cfg)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_a)

        self.critic = Critic(self.obs_dim, self.action_dim, **net_cfg)
        self.critic_target = Critic(self.obs_dim, self.action_dim, **net_cfg)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_c)

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.gamma

        #
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True
        self.max_action = args.max_action

        if USE_CUDA: self.cuda()

    def update_policy(self, replay_buffer):

        # Sample batch
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = replay_buffer.sample()

        # Prepare for the target q batch
        next_q_values = self.critic_target([
            to_tensor(next_obs_batch, volatile=True),
            self.actor_target(to_tensor(next_obs_batch, volatile=True)),
        ])
        next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
                         self.discount * to_tensor(done_batch) * next_q_values

        # Critic update
        self.critic.zero_grad()

        q_batch = self.critic([to_tensor(obs_batch), to_tensor(action_batch)])

        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        policy_loss = -self.critic([
            to_tensor(obs_batch),
            self.actor(to_tensor(obs_batch))
        ])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def eval(self):
        self.evaluation = True
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def random_action(self):
        action = np.random.uniform(-self.max_action, self.max_action, self.action_dim)
        return action

    def select_action(self, obs, noise_std):
        action = to_numpy(
            self.actor(to_tensor(np.array([obs])))
        ).squeeze(0)
        action = (action + np.random.normal(0, noise_std, size=self.action_dim)).clip(-self.max_action, self.max_action)
        return action

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
