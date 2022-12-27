import math
import random
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree


class SSLShootEnv(SSLBaseEnv):
    """The SSL robot needs to make a goal on a field with static defenders

        Description:

        Observation:
            Type: Box(4 + 8*n_robots_blue + 2*n_robots_yellow)
            Normalized Bounds to [-1.2, 1.2]
            Num      Observation normalized
            0->3     Ball [X, Y, V_x, V_y]
            4+i*7->10+i*7    id i(0-2) Blue [X, Y, sin(theta), cos(theta), v_x, v_y, v_theta]
            24+j*5,25+j*5     id j(0-2) Yellow Robot [X, Y, v_x, v_y, v_theta]
            41      Possession Player None -1 Blue 0-2 Yellow 3-5
            42      Active Player 0-2
        Actions:
            Type: Box(5, )
            Num     Action
            0       id 0 Blue Global X Direction Speed  (%)
            1       id 0 Blue Global Y Direction Speed  (%)
            2       id 0 Blue Angular Speed  (%)
            3       id 0 Blue Kick x Speed  (%)
            4       id 0 Blue Dribbler  (%) (true if % is positive)

        Reward:
            1 if goal
        Starting State:
            Robot on field center, ball and defenders randomly positioned on
            positive field side
        Episode Termination:
            Goal, 25 seconds (1000 steps), or rule infraction
    """

    def __init__(self, field_type=2):
        super().__init__(field_type=field_type, n_robots_blue=3,
                         n_robots_yellow=3, time_step=0.025)

        self.max_dribbler_time = 100
        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(5,), dtype=np.float32)

        n_obs = 4 + 7 * self.n_robots_blue + 5 * self.n_robots_yellow + 2
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(n_obs,),
                                                dtype=np.float32)

        # scale max dist rw to 1 Considering that max possible move rw if ball and robot are in opposite corners of field
        self.ball_dist_scale = np.linalg.norm([self.field.width, self.field.length / 2])
        self.ball_grad_scale = np.linalg.norm([self.field.width / 2, self.field.length / 2]) / 4

        # scale max energy rw to 1 Considering that max possible energy if max robot wheel speed sent every step
        wheel_max_rad_s = 160
        max_steps = 1000
        self.energy_scale = ((wheel_max_rad_s * 4) * max_steps)
        self.previous_ball_potential = None

        # Limit robot speeds
        self.max_v = 2.5
        self.max_w = 10
        self.kick_speed_x = 5.0

        self.active_robot_idx = 0
        self.possession_robot_idx = -1

        self.dribbler_time = 0
        self.commands = None
        print('Environment initialized', "Obs:", n_obs)

    def reset(self):
        self.reward_shaping_total = None
        self.dribbler_time = 0
        return super().reset()

    def step(self, action):
        self.reward_shaping_total = None
        observation, reward, done, _ = super().step(action)

        return observation, reward, done, self.reward_shaping_total

    def _frame_to_observations(self):

        observation = []

        observation.append(self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        for i in range(self.n_robots_blue):
            observation.append(self.norm_pos(self.frame.robots_blue[i].x))
            observation.append(self.norm_pos(self.frame.robots_blue[i].y))
            observation.append(
                np.sin(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(
                np.cos(np.deg2rad(self.frame.robots_blue[i].theta))
            )
            observation.append(self.norm_v(self.frame.robots_blue[i].v_x))
            observation.append(self.norm_v(self.frame.robots_blue[i].v_y))
            observation.append(self.norm_w(self.frame.robots_blue[i].v_theta))
            # observation.append(1 if self.frame.robots_blue[i].infrared else 0)

        for i in range(self.n_robots_yellow):
            observation.append(self.norm_pos(self.frame.robots_yellow[i].x))
            observation.append(self.norm_pos(self.frame.robots_yellow[i].y))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_x))
            observation.append(self.norm_v(self.frame.robots_yellow[i].v_y))
            observation.append(self.norm_w(self.frame.robots_yellow[i].v_theta))

        nearest_blue_robot, nearest_blue_robot_dist = self.get_nearest_robot_idx(
            [self.frame.ball.x, self.frame.ball.y], "blue")
        nearest_yellow_robot, nearest_yellow_robot_dist = self.get_nearest_robot_idx(
            [self.frame.ball.x, self.frame.ball.y], "yellow")

        threshold = 0.12

        last_active = self.active_robot_idx
        last_possession = self.possession_robot_idx

        self.active_robot_idx = nearest_blue_robot
        self.possession_robot_idx = -1
        if self.commands is not None:
            if nearest_blue_robot_dist <= nearest_yellow_robot_dist:
                if nearest_blue_robot_dist <= threshold and self.commands[
                    nearest_blue_robot].dribbler and self.__is_toward_ball("blue", nearest_blue_robot):
                    self.possession_robot_idx = nearest_blue_robot
            else:
                if nearest_yellow_robot_dist <= threshold and self.commands[
                    self.n_robots_blue + nearest_yellow_robot].dribbler and self.__is_toward_ball("yellow",
                                                                                                  nearest_blue_robot):
                    self.possession_robot_idx = 3 + nearest_yellow_robot

            if self.possession_robot_idx == last_possession and self.possession_robot_idx == self.active_robot_idx:
                self.dribbler_time += 1
            else:
                self.dribbler_time = 0

        observation.append(self.possession_robot_idx)
        observation.append(self.active_robot_idx)

        # print("possession_robot:", self.possession_robot_idx)
        # print("active_robot:", self.active_robot_idx)

        return np.array(observation, dtype=np.float32)

    def _get_commands(self, actions):
        commands = []

        # Blue robot
        for i in range(self.n_robots_blue):
            if i != self.active_robot_idx:
                actions = self.action_space.sample()
                angle = self.frame.robots_blue[i].theta
                v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(angle))
                cmd = Robot(yellow=False, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta,
                            kick_v_x=self.kick_speed_x if actions[3] > 0 else 0.,
                            dribbler=True if actions[4] > 0 else False)
                commands.append(cmd)
            else:
                # Controlled robot
                angle = self.frame.robots_blue[self.active_robot_idx].theta
                v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(angle))
                cmd = Robot(yellow=False, id=0, v_x=v_x, v_y=v_y, v_theta=v_theta,
                            kick_v_x=self.kick_speed_x if actions[3] > 0 else 0.,
                            dribbler=True if actions[4] > 0 else False)
                commands.append(cmd)

        # Yellow robot
        for i in range(self.n_robots_yellow):
            actions = self.action_space.sample()
            angle = self.frame.robots_yellow[i].theta
            v_x, v_y, v_theta = self.convert_actions(actions, np.deg2rad(angle))
            cmd = Robot(yellow=True, id=i, v_x=v_x, v_y=v_y, v_theta=v_theta,
                        kick_v_x=self.kick_speed_x if actions[3] > 0 else 0.,
                        dribbler=True if actions[4] > 0 else False)
            commands.append(cmd)
        self.commands = commands
        return commands

    def convert_actions(self, action, angle):
        """Denormalize, clip to absolute max and convert to local"""
        # Denormalize
        v_x = action[0] * self.max_v
        v_y = action[1] * self.max_v
        v_theta = action[2] * self.max_w
        # Convert to local
        v_x, v_y = v_x * np.cos(angle) + v_y * np.sin(angle), \
                   -v_x * np.sin(angle) + v_y * np.cos(angle)

        # clip by max absolute
        v_norm = np.linalg.norm([v_x, v_y])
        c = v_norm < self.max_v or self.max_v / v_norm
        v_x, v_y = v_x * c, v_y * c

        return v_x, v_y, v_theta

    def _calculate_reward_and_done(self):
        if self.reward_shaping_total is None:
            self.reward_shaping_total = {
                'goal': 0,
                'rbt_in_gk_area': 0,
                'done_ball_out': 0,
                'done_ball_out_right': 0,
                'done_rbt_out': 0,
                'done_long_dribbler': 0,
                'rw_ball_grad': 0,
                'rw_towards_ball': 0,
                'rw_dribble': 0,
                'rw_energy': 0
            }
        reward = 0
        done = False

        # Field parameters
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2
        half_goal_wid = self.field.goal_width / 2

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]

        def robot_in_gk_area(rbt):
            return rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid

        if self.dribbler_time > self.max_dribbler_time:
            done = True
            self.reward_shaping_total['done_long_dribbler'] += 1
            reward = -1
        # Check if robot exited field right side limits
        if robot.x < -0.2 or abs(robot.y) > half_wid:
            done = True
            self.reward_shaping_total['done_rbt_out'] += 1
            reward = -1
        # # If flag is set, end episode if robot enter gk area
        # elif robot_in_gk_area(robot):
        #     done = True
        #     self.reward_shaping_total['rbt_in_gk_area'] += 1
        # Check ball for ending conditions
        elif ball.x < 0 or abs(ball.y) > half_wid:
            done = True
            self.reward_shaping_total['done_ball_out'] += 1
            reward = -1
        elif ball.x > half_len:
            done = True
            if abs(ball.y) < half_goal_wid:
                reward = 10
                self.reward_shaping_total['goal'] += 1
            else:
                reward = 0
                self.reward_shaping_total['done_ball_out_right'] += 1
        elif self.last_frame is not None:
            # ball_dist_rw = self.__ball_dist_rw() / self.ball_dist_scale
            # self.reward_shaping_total['ball_dist'] += ball_dist_rw

            ball_grad_rw = self.__ball_grad_rw() / self.ball_grad_scale
            self.reward_shaping_total['rw_ball_grad'] += ball_grad_rw

            toward_ball_rw = self.__towards_ball_rw()
            self.reward_shaping_total['rw_towards_ball'] += toward_ball_rw

            dribble_shoot_rw = self.__dribble_shoot_rw(toward_ball_rw)
            self.reward_shaping_total['rw_dribble'] += dribble_shoot_rw

            energy_rw = -self.__energy_pen() / self.energy_scale
            self.reward_shaping_total['rw_energy'] += energy_rw

            reward = reward + ball_grad_rw + 0.5 * toward_ball_rw + 0.2 * dribble_shoot_rw + energy_rw

        done = done
        return reward, done

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2

        def x():
            return random.uniform(-half_len+0.1, half_len-0.1)

        def y():
            return random.uniform(-half_wid + 0.1, half_wid - 0.1)

        def theta():
            return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.robots_blue[0] = Robot(x=0., y=0., theta=0.)

        def in_gk_area(obj):
            return obj.x > half_len - pen_len and abs(obj.y) < half_pen_wid

        pos_frame.ball = Ball(x=x(), y=y())
        while in_gk_area(pos_frame.ball):
            pos_frame.ball = Ball(x=x(), y=y())
        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))

        factor = (pos_frame.ball.y / abs(pos_frame.ball.y))
        offset = 0.115 * factor
        angle = random.uniform(0, 360)
        pos_frame.robots_blue[0] = Robot(
            x=pos_frame.ball.x, y=pos_frame.ball.y + offset, theta=angle
        )
        places.insert((pos_frame.ball.x, pos_frame.ball.y + offset))
        min_dist = 0.2
        for i in range(self.n_robots_blue):
            if i == 0:
                continue
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def __ball_dist_rw(self):
        assert (self.last_frame is not None)

        # Calculate previous ball dist
        last_ball = self.last_frame.ball
        last_robot = self.last_frame.robots_blue[0]
        last_ball_pos = np.array([last_ball.x, last_ball.y])
        last_robot_pos = np.array([last_robot.x, last_robot.y])
        last_ball_dist = np.linalg.norm(last_robot_pos - last_ball_pos)

        # Calculate new ball dist
        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        ball_pos = np.array([ball.x, ball.y])
        robot_pos = np.array([robot.x, robot.y])
        ball_dist = np.linalg.norm(robot_pos - ball_pos)

        ball_dist_rw = last_ball_dist - ball_dist

        if ball_dist_rw > 1:
            print("ball_dist -> ", ball_dist_rw)
            print(self.frame.ball)
            print(self.frame.robots_blue)
            print(self.frame.robots_yellow)
            print("===============================")

        return np.clip(ball_dist_rw, -1, 1)

    def __ball_grad_rw(self):
        '''Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        '''
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0) \
                      + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step,
                                          -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def __energy_pen(self):
        robot = self.frame.robots_blue[0]

        # Sum of abs each wheel speed sent
        energy = abs(robot.v_wheel0) \
                 + abs(robot.v_wheel1) \
                 + abs(robot.v_wheel2) \
                 + abs(robot.v_wheel3)

        return energy

    def __towards_ball_rw(self):
        theta = math.radians(self.frame.robots_blue[0].theta)
        Xr, Yr, theta, Xb, Yb = self.frame.robots_blue[0].x, self.frame.robots_blue[
            0].y, theta, self.frame.ball.x, self.frame.ball.y

        # 计算机器人-球连线方向的角度
        line_angle = math.atan2(Yb - Yr, Xb - Xr)

        # 计算摄像头方向和机器人-球连线方向之间的夹角
        angle = line_angle - theta

        # 将夹角转换为[-π,π]区间内的值
        while angle < -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi

        value = math.pi - abs(angle)
        normalized_value = (value - 0) / (math.pi - 0)
        if normalized_value > 0.9:
            normalized_value = 1
        return normalized_value

    def __is_toward_ball(self, team, idx):
        if team == "blue":
            Xr, Yr = self.frame.robots_blue[idx].x, self.frame.robots_blue[idx].y
            theta = math.radians(self.frame.robots_blue[idx].theta)
        if team == "yellow":
            theta = math.radians(self.frame.robots_yellow[idx].theta)
            Xr, Yr = self.frame.robots_yellow[idx].x, self.frame.robots_yellow[idx].y
        Xb, Yb = self.frame.ball.x, self.frame.ball.y

        # 计算机器人-球连线方向的角度
        line_angle = math.atan2(Yb - Yr, Xb - Xr)

        # 计算摄像头方向和机器人-球连线方向之间的夹角
        angle = line_angle - theta

        # 将夹角转换为[-π,π]区间内的值
        while angle < -math.pi:
            angle += 2 * math.pi
        while angle > math.pi:
            angle -= 2 * math.pi

        value = math.pi - abs(angle)
        normalized_value = (value - 0) / (math.pi - 0)
        if normalized_value >= 0.9:
            return True
        else:
            return False

    def __dribble_shoot_rw(self, toward_ball_rw):
        rbt = self.frame.robots_blue[self.active_robot_idx]
        if self.possession_robot_idx == self.active_robot_idx:
            if self.frame.robots_blue[self.active_robot_idx].kick_v_x >=0 or self.frame.robots_blue[self.active_robot_idx].kick_v_z >=0:
                return 0.2
            else:
                return 1
        else:
            return 0