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
        self.noise_std = self.args.noise_std_init
        print(f"action_dim={self.args.action_dim}")
        print(f"obs_dim={self.args.obs_dim}")


        # Create a tensorboard
        self.writer = None
        if not self.args.display:
            self.writer = SummaryWriter(
                log_dir=f'./runs/{self.args.env_name}/{number}')

        # Create agent and buffer
        self.agent = DDPG(args)
        if args.restore:
            self.agent.actor.load_state_dict(torch.load(args.restore_ckp_actor))
            self.agent.critic.load_state_dict(torch.load(args.restore_ckp_critic))
            print("Successfully load ckp from: ",args.restore_ckp_actor)
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
                action = self.agent.select_action(obs,self.noise_std)
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

                # Train agent
                if self.replay_buffer.current_size > self.args.batch_size and not self.args.display:
                    self.agent.update_policy(self.replay_buffer)

                # Decay noise_std
                self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if episode_step >= self.args.episode_limit:
                    terminate = True

            self.episode += 1
            if args.write_rewards:
                for r in reward_dict:
                    self.writer.add_scalar(r, reward_dict[r]/ episode_step, global_step=self.episode)

            # Save model
            if self.episode % self.args.save_rate == 0 and not self.args.display:
                save_dir = f"./models/{self.args.env_name}/{number}"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.agent.actor.state_dict(),
                           f"{save_dir}/{self.args.env_name}_{number}_{int(self.total_steps / 1000)}k_actor.npy")
                torch.save(self.agent.critic.state_dict(),
                           f"{save_dir}/{self.args.env_name}_{number}_{int(self.total_steps / 1000)}k_critic.npy")

            avg_train_reward = episode_reward / episode_step
            print("============epi={},step={},avg_reward={},goal_score={},noise={}==============".format(self.episode,
                                                                                                self.total_steps,
                                                                                                avg_train_reward,
                                                                                                info["goal"],
                                                                                                self.noise_std))
            print("last action: ",action)
            if not self.args.display:
                self.writer.add_scalar('avg_episode_reward', avg_train_reward, global_step=self.episode)
                self.writer.add_scalar('goal', info["goal"], global_step=self.episode)
        self.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for individual models.")
    parser.add_argument("--env_name", type=str, default="SSLShootEnv", help="Environemnt name.")
    parser.add_argument("--max_episode", type=int, default=int(200000), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=300, help="Maximum number of steps per episode")
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")
    parser.add_argument("--buffer_size", type=int, default=int(1e7), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim_1", type=int, default=256,
                        help="The number of neurons in 1st hidden layers of the neural network")
    parser.add_argument("--hidden_dim_2", type=int, default=256,
                        help="The number of neurons in 2rd hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=1e6,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--write_rewards", type=bool, default=True, help="Whether to write reward")
    parser.add_argument("--lr_a", type=float, default=1e-5, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-5, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    # parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--save_rate", type=int, default=2000,
                        help="Model save per n episode")
    parser.add_argument("--policy_update_freq", type=int, default=1, help="The frequency of policy updates")
    parser.add_argument("--display", type=bool, default=False, help="Display mode")
    parser.add_argument("--restore", type=bool, default=False, help="")
    parser.add_argument("--restore_ckp_actor", type=str, default="models/SSLShootEnv/38/SSLShootEnv_38_2016k_actor.npy", help="")
    parser.add_argument("--restore_ckp_critic", type=str, default="models/SSLShootEnv/38/SSLShootEnv_38_2016k_critic.npy", help="")
    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    number = 40
    runner = Runner(args, number=number)

    # Save args
    save_dir = f"./models/args"
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}/args_num_{number}.npy', 'wb') as f:
        pickle.dump(runner.args, f)

    print("Start runner.run()")
    runner.run()
