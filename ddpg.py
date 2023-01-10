import numpy as np
import numpy.random
import torch
import torch.nn as nn
from torch.optim import Adam

from utils import *

criterion = nn.MSELoss()


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.tanh(out)
        return out


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=400, hidden2=300, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1 + nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        orthogonal_init(self.fc1)
        orthogonal_init(self.fc2)
        orthogonal_init(self.fc3)

    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)

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

        self.random_process = OrnsteinUhlenbeckProcess(size=self.action_dim, theta=args.ou_theta, mu=args.ou_mu,
                                                       sigma=args.ou_sigma)
        # Hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.gamma

        #
        self.s_t = None  # Most recent state
        self.a_t = None  # Most recent action
        self.is_training = True
        self.max_action = args.max_action

        # Noise
        self.epsilon = args.epsilon_init
        self.epsilon_min = args.epsilon_min
        self.epsilon_decay = args.epsilon_decay
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
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        action = to_numpy(
            self.actor(to_tensor(np.array([s_t])))
        ).squeeze(0)
        if numpy.random.rand() < self.epsilon and self.evaluation is False:
            action = self.random_process.sample()
        action = np.clip(action, -self.max_action, self.max_action)

        # Decay noise_std
        if decay_epsilon and self.evaluation is False:
            self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_min)

        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
