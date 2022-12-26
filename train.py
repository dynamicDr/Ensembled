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
        self.seed = 0
        # Create env
        self.env: VSSMAEnv = gym.make(self.args.env_name)
        self.args.N = 1  # The number of agents
        self.args.obs_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.shape[0]
        print(f"action_dim={self.args.action_dim}")
        print(f"obs_dim={self.args.obs_dim}")

        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Create a tensorboard
        self.writer = None
        if not self.args.display:
            self.writer = SummaryWriter(
                log_dir=f'./runs/{self.args.env_name}/{number}')

        # Create agent and buffer
        self.agent = DDPG(args, self.writer)
        self.replay_buffer = ReplayBuffer(self.args)

    def run(self):
        while self.episode < self.args.max_episode:
            # For each episode..
            obs = self.env.reset()
            terminate = False
            done = False
            episode_step = 0
            episode_reward = 0
            while not (done or terminate):
                # For each step...
                action = self.agent.select_action(obs, noise_std=self.noise_std)
                obs_next, reward, done, info = self.env.step(copy.deepcopy(action))
                if self.args.display:
                    self.env.render()
                # Store the transition
                self.replay_buffer.store_transition(obs, action, reward, obs_next, done)
                obs = obs_next
                self.total_steps += 1
                episode_step += 1
                episode_reward += reward

                # Decay noise_std
                if self.args.use_noise_decay:
                    self.noise_std = self.noise_std - self.args.noise_std_decay if self.noise_std - self.args.noise_std_decay > self.args.noise_std_min else self.args.noise_std_min

                if self.replay_buffer.current_size > self.args.batch_size and not self.args.display:
                    # Train agent
                    self.agent.train(self.replay_buffer)

                if episode_step >= self.args.episode_limit:
                    terminate = True

            self.episode += 1

            # Save model
            if self.episode % self.args.save_rate == 0 and not self.args.display:
                save_dir = f"./model/{self.args.env_name}/{number}"
                os.makedirs(save_dir, exist_ok=True)
                torch.save(self.agent.actor.state_dict(),
                           f"{dir}/{self.args.env_name}_{number}_{int(self.total_steps / 1000)}_actor.npy")
                torch.save(self.agent.actor.state_dict(),
                           f"{dir}/{self.args.env_name}_{number}_{int(self.total_steps / 1000)}_critic.npy")

            avg_train_reward = episode_reward / episode_step
            print("============epi={},step={},avg_reward={},goal_score={}==============".format(self.episode,
                                                                                                self.total_steps,
                                                                                                avg_train_reward,
                                                                                                info["goal_score"]))
            if not self.args.display:
                self.writer.add_scalar('avg_episode_reward', avg_train_reward, global_step=self.episode)
                self.writer.add_scalar('goal', info["goal_score"], global_step=self.episode)
        self.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for individual models.")
    parser.add_argument("--env_name", type=str, default="SSLShootEnv", help="Environemnt name.")
    parser.add_argument("--max_episode", type=int, default=int(21000), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=300, help="Maximum number of steps per episode")  # 300
    parser.add_argument("--max_action", type=float, default=1.0, help="Max action")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="The capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--noise_std_init", type=float, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_std_min", type=float, default=0.05, help="The std of Gaussian noise for exploration")
    parser.add_argument("--noise_decay_steps", type=float, default=3e5,
                        help="How many steps before the noise_std decays to the minimum")
    parser.add_argument("--use_noise_decay", type=bool, default=True, help="Whether to decay the noise_std")
    parser.add_argument("--lr_a", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=1e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="Softly update the target network")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--save_rate", type=int, default=1000,
                        help="Model save per n episode")
    parser.add_argument("--policy_noise", type=float, default=0.2, help="Target policy smoothing")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="Clip noise")
    parser.add_argument("--policy_update_freq", type=int, default=2, help="The frequency of policy updates")
    parser.add_argument("--display", type=bool, default=False, help="Display mode")
    # parser.add_argument("--restore", type=bool, default=False, help="Restore from checkpoint")
    # parser.add_argument("--restore_episode", type=int, default=16000, help="Restore from checkpoint")
    # parser.add_argument("--restore_step", type=int, default=40940000, help="Restore from checkpoint")
    #
    # parser.add_argument("--restore_model_dir", type=str,
    #                     default="/home/user/football/HRL/models/agent/actor_number_19_552k_agent_{}.pth",
    #                     help="Restore from checkpoint")
    args = parser.parse_args()
    args.noise_std_decay = (args.noise_std_init - args.noise_std_min) / args.noise_decay_steps

    number = 1
    runner = Runner(args, number=number)

    # Save args
    with open(f'./models/args/args_num_{number}.npy', 'wb') as f:
        pickle.dump(runner.args, f)

    print("Start runner.run()")
    runner.run()
