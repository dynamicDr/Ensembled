import argparse
import copy
import pickle
import re

import gym
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from ddpg import DDPG
from replay_buffer import ReplayBuffer
from rsoccer_gym.vss.env_ma import VSSMAEnv


class Runner:
    def __init__(self, args, number):

        self.total_steps = 0
        self.episode = 0
        self.args = args
        self.number = number
        # Create env
        self.env = gym.make(self.args.env_name+"-v0")
        self.args.N = 1  # The number of agents
        self.args.obs_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.shape[0]
        print(f"action_dim={self.args.action_dim}")
        print(f"obs_dim={self.args.obs_dim}")


        # Create a tensorboard
        self.writer = None
        if not self.args.display:
            self.writer = SummaryWriter(
                log_dir=f'./runs/{self.args.env_name}/{number}')

        # Create agent and buffer
        self.agent = DDPG(args)
        self.replay_buffer = ReplayBuffer(self.args)

    def run(self):
        reward_dict = {}
        while self.episode < self.args.max_episode:
            # For each episode..
            obs = self.env.reset()
            for r in reward_dict:
                reward_dict[r] = 0
            terminate = False
            done = False
            episode_step = 0
            episode_reward = 0
            while not (done or terminate):
                # For each step...
                action = self.agent.select_action(obs)
                obs_next, reward, done, info = self.env.step(copy.deepcopy(action))
                if args.write_rewards:
                    for r in info:
                        if r.startswith("rw_"):
                            if r not in reward_dict:
                                reward_dict[r] = 0
                            reward_dict[r] += info[r]
                if self.args.display:
                    self.env.render()

                # Store the transition
                self.replay_buffer.store_transition(obs, action, reward, obs_next, done)
                obs = obs_next
                self.total_steps += 1
                episode_step += 1
                episode_reward += reward


                if self.replay_buffer.current_size > self.args.batch_size and not self.args.display:
                    # Train agent
                    self.agent.update_policy(self.replay_buffer)

                if episode_step >= self.args.episode_limit:
                    terminate = True

            self.episode += 1
            if args.write_rewards:
                for r in reward_dict:
                    self.writer.add_scalar(r, reward_dict[r], global_step=self.episode)

            # Save model
            if self.episode % self.args.save_rate == 0 and not self.args.display:
                save_dir = f"./models/{self.args.env_name}/{number}"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.agent.actor.state_dict(),
                           f"{save_dir}/{self.args.env_name}_{number}_{int(self.total_steps / 1000)}k_actor.npy")
                torch.save(self.agent.critic.state_dict(),
                           f"{save_dir}/{self.args.env_name}_{number}_{int(self.total_steps / 1000)}k_critic.npy")

            avg_train_reward = episode_reward / episode_step
            print("============epi={},step={},avg_reward={},goal_score={},epsilon={}==============".format(self.episode,
                                                                                                self.total_steps,
                                                                                                avg_train_reward,
                                                                                                info["goal"],
                                                                                                self.agent.epsilon))
            if not self.args.display:
                self.writer.add_scalar('avg_episode_reward', avg_train_reward, global_step=self.episode)
                self.writer.add_scalar('goal', info["goal"], global_step=self.episode)
        self.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for individual models.")
    parser.add_argument("--env_name", type=str, default="SSLShootEnv", help="Environemnt name.")
    parser.add_argument("--max_episode", type=int, default=int(200000), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=500, help="Maximum number of steps per episode")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim_1", type=int, default=32,
                        help="The number of neurons in 1st hidden layers of the neural network")
    parser.add_argument("--hidden_dim_2", type=int, default=16,
                        help="The number of neurons in 2rd hidden layers of the neural network")
    parser.add_argument("--epsilon_init", type=float, default=1, help="The std of Gaussian noise for exploration")
    parser.add_argument("--epsilon_min", type=float, default=0.01, help="The std of Gaussian noise for exploration")
    parser.add_argument("--epsilon_decay_steps", type=float, default=5e5,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--write_rewards", type=bool, default=True, help="Whether to write reward")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    # parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--save_rate", type=int, default=1000,
                        help="Model save per n episode")
    parser.add_argument("--policy_update_freq", type=int, default=1, help="The frequency of policy updates")
    parser.add_argument("--display", type=bool, default=False, help="Display mode")
    args = parser.parse_args()
    args.epsilon_decay = (args.epsilon_init - args.epsilon_min) / args.epsilon_decay_steps

    number = 5
    runner = Runner(args, number=number)

    # Save args
    save_dir = f"./models/args"
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/args_num_{number}.npy', 'wb') as f:
        pickle.dump(runner.args, f)

    print("Start runner.run()")
    runner.run()
