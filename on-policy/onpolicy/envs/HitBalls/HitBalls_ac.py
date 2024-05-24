from gym import spaces
from onpolicy.config import get_config
import numpy as np
import yaml
import copy
import time
import torch
import threading

from onpolicy.utils.plot3d_test import plot_track, plot_gif_plankNball
from onpolicy.envs.HitBalls.Plank_ball_policy import Policy_P2B_straight, Policy_P2B_rotate
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as multi_policy
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

def _t2n(x):
    return x.detach().cpu().numpy()

def norm_batch(raw_mat):
    norms = np.linalg.norm(raw_mat, axis=-1)
    mask = norms > 1
    raw_mat[mask, :] = raw_mat[mask, :] / norms[mask, np.newaxis]
    return raw_mat

def elementwise_and(arrays):
    result = np.array(arrays[0])
    for arr in arrays[1:]:
        result = np.logical_and(result, arr)
    return result

def elementwise_or(arrays):
    result = np.array(arrays[0])
    for arr in arrays[1:]:
        result = np.logical_or(result, arr)
    return result

def limit_to_pi_half(arr):
    arr = np.array(arr)
    arr = arr % (np.pi)
    arr[arr > np.pi/2] -= np.pi  
    return arr

def elastic_collision(plank_state, ball_state):
    plank_pos = plank_state['position']
    plank_vel = plank_state['velocity']
    plank_theta = plank_state['orientation']
    plank_heading = plank_state['bodyrate']
    plank_length = plank_state['plank_length']
    bounce_width = plank_state['bounce_width']

    ball_pos = ball_state['position']
    ball_vel = ball_state['velocity']


    line_equation = np.cos(plank_theta) * (ball_pos[:, 1:2] - plank_pos[:, 1:2]) - np.sin(plank_theta) * (ball_pos[:, :1] - plank_pos[:, :1])
    normal_line_equation = np.cos(plank_theta) * (ball_pos[:, :1] - plank_pos[:, :1]) + np.sin(plank_theta) * (ball_pos[:, 1:2] - plank_pos[:, 1:2])
    quadrant_line = np.sign(line_equation)
    normal_quad_line = np.sign(normal_line_equation)
    dis_ball2plank_n = np.abs(line_equation)
    dis_ball2plank_l = np.abs(normal_line_equation)

    mask_collision = np.logical_and(dis_ball2plank_n < bounce_width, dis_ball2plank_l < plank_length / 2)

    normal_vel = plank_vel[:,1:2] * np.cos(plank_theta) - plank_vel[:,:1] * np.sin(plank_theta) + normal_quad_line * dis_ball2plank_l * plank_heading
    ball_normal_vel = ball_vel[:,1:2] * np.cos(plank_theta) - ball_vel[:,:1] * np.sin(plank_theta)
    ball_tan_vel = ball_vel[:,:1] * np.cos(plank_theta) + ball_vel[:,1:2] * np.sin(plank_theta)

    delta_vel =  (ball_normal_vel - normal_vel)

    mask_collision = np.logical_and(mask_collision, (quadrant_line * delta_vel) < 0)
    ball_normal_vel[mask_collision] = ball_normal_vel[mask_collision] - 2 * delta_vel[mask_collision]
    ball_vel_new = np.zeros_like(ball_vel)
    ball_vel_new[:,:1] = ball_tan_vel * np.cos(plank_theta) - ball_normal_vel * np.sin(plank_theta)
    ball_vel_new[:,1:2] = ball_tan_vel * np.sin(plank_theta) + ball_normal_vel * np.cos(plank_theta)

    return ball_vel_new




class PlankAndBall:
    def __init__(self, num_threads, cfg_path = None, oppo_policy=None):
        self.num_threads = num_threads

        if cfg_path is None:
            cfg_path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/envs/HitBalls/config_ac.yaml'
        self.cfg_ = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

        self.config = self.cfg_['ball_env']

        self.episode_length = self.config['episode_length']
        self.step_counter = np.zeros(num_threads)
        self.ongoing = np.zeros(num_threads, dtype=bool)

        # agent configs
        self.agent_cfg = self.config['agents']
        self.sim_dt = self.agent_cfg['sim_dt']
        self.max_vel_ball = self.agent_cfg['max_vel_ball']
        self.max_vel_plank = self.agent_cfg['max_vel_plank']
        self.max_acc_line_plank = self.agent_cfg['max_acc_line_plank']
        self.max_acc_angle_plank = self.agent_cfg['max_acc_angle_plank']
        self.init_vel_ball = self.agent_cfg['init_ball_vel']
        self.max_heading_plank = self.agent_cfg['max_heading_rate']
        self.plank_length = self.agent_cfg['plank_length']
        self.bounce_width = self.agent_cfg['bounce_width']
        self.ball_radius = self.agent_cfg['ball_radius']

        # env configs
        self.world_box = self.config['env']['world_box']
        self.lose_zone_x = self.config['env']['lose_zone_x']
        self.mid_zone_x = self.config['env']['mid_zone_x']

        # reward configs
        self.reward_coef = self.config['reward']
        self.bounce_ball_coef = self.reward_coef['bounce_ball_coef']
        self.win_coef = self.reward_coef['win_coef']
        self.lose_coef = self.reward_coef['lose_coef']
        self.step_coef = self.reward_coef['step_coef']
        self.delta_action_coef = self.reward_coef['delta_action_coef']
        self.time_discount_coef = self.reward_coef['time_discount_coef']

        # observation configs
        self.ob_config = self.config['observation']
        self.prop_obs = self.ob_config['proprioceptive']
        self.ex_obs = self.ob_config['exterprioceptive']
        self.enemy_ob = self.ex_obs['enemy']
        self.ball_ob = self.ex_obs['ball']

        self.obs_map = dict()
        self.obs_map['position'] = 2
        self.obs_map['orientation'] = 1
        self.obs_map['velocity'] = 2
        self.obs_map['bodyrate'] = 1

        print("envs world box = ", self.world_box)

        observation_length = 0

        for term in self.prop_obs:
            observation_length += self.obs_map[term]
        for term in self.enemy_ob:
            observation_length += self.obs_map[term]
        for term in self.ball_ob:
            observation_length += self.obs_map[term]

        # set spaces   
        self.observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(observation_length + 1,), dtype=np.float32) for _ in range(1)]
        self.share_observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(observation_length +1,), dtype=np.float32) for _ in range(1)]
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) for _ in range(1)]

        self.oppo_obs_space = [spaces.Box(low=-1.0, high=1.0, shape=(observation_length + 1,), dtype=np.float32) for _ in range(1)]
        self.oppo_act_space = [spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32) for _ in range(1)]


        self.oppo_policy = copy.deepcopy(oppo_policy)
        # self.obs = np.zeros((self.num_threads, self.num_role[self.team_name], 3*self.total_num_agents))
        # self.oppo_obs = np.zeros((self.num_threads, self.num_role[self.oppo_name], 3*self.total_num_agents))
        self.obs_all_raw = np.zeros((self.num_threads, 16))
        self.rewards = np.zeros((self.num_threads, 1))
        self.dones = np.zeros((self.num_threads, 1), dtype=bool)
        self.oppo_done = np.zeros((self.num_threads, 1), dtype=bool)
        self.act_record = np.zeros((self.num_threads, 2*3))
        self.act_prev = np.zeros((self.num_threads, 2*3))

        seed = int((int(time.time()*100) % 10000) * 1000)
        self.rng = np.random.default_rng(seed)

    def seed(self, seed=None):
        if seed is None:
            self.rng = np.random.default_rng(1)
        else:
            self.rng = np.random.default_rng(seed)

    def gen_obs_from_raw(self):
        # obs_raw: [plank1: {pos, angle, vel, heading},
        #           plank2: {pos, angle, vel, heading},
        #           ball: {pos, vel}]
        time_ = self.step_counter * self.sim_dt
        self.obs = np.concatenate([self.obs_all_raw[:,:9], self.obs_all_raw[:,12:], time_[:, np.newaxis]], axis= -1)
        oppo_self = np.concatenate([- self.obs_all_raw[:,6:8], self.obs_all_raw[:,8:9], -self.obs_all_raw[:,9:11], self.obs_all_raw[:,11:12]], axis=-1)
        oppo_others = np.concatenate([- self.obs_all_raw[:, 0:2], self.obs_all_raw[:,2:3], -self.obs_all_raw[:, 12:]], axis=-1)
        self.oppo_obs = np.concatenate([oppo_self, oppo_others, time_[:, np.newaxis]], axis= -1)
        self.obs = self.obs[:, np.newaxis, :]
        self.oppo_obs = self.oppo_obs[:, np.newaxis, :]

    def constrain_terms(self):
        self.obs_all_raw[:, 2:3] = limit_to_pi_half(self.obs_all_raw[:, 2:3])
        self.obs_all_raw[:, 8:9] = limit_to_pi_half(self.obs_all_raw[:, 8:9])

        plank1_left_pos = self.obs_all_raw[:,:1] - np.cos(self.obs_all_raw[:,2:3]) * self.plank_length / 2
        plank1_right_pos = self.obs_all_raw[:,:1] + np.cos(self.obs_all_raw[:,2:3]) * self.plank_length / 2
        plank2_left_pos = self.obs_all_raw[:,6:7] - np.cos(self.obs_all_raw[:,8:9]) * self.plank_length / 2
        plank2_right_pos = self.obs_all_raw[:,6:7] + np.cos(self.obs_all_raw[:,8:9]) * self.plank_length / 2
        mask_left_bound_p1 = plank1_left_pos < self.world_box[2]
        mask_left_bound_p2 = plank2_left_pos < self.world_box[2]
        mask_right_bound_p1 = plank1_right_pos > self.world_box[3]
        mask_right_bound_p2 = plank2_right_pos > self.world_box[3]
        self.obs_all_raw[:,:1][mask_left_bound_p1] -= plank1_left_pos[mask_left_bound_p1] - self.world_box[2]
        self.obs_all_raw[:,6:7][mask_left_bound_p2] -= plank2_left_pos[mask_left_bound_p2] - self.world_box[2]
        self.obs_all_raw[:,:1][mask_right_bound_p1] -= plank1_right_pos[mask_right_bound_p1] - self.world_box[3]
        self.obs_all_raw[:,6:7][mask_right_bound_p2] -= plank2_right_pos[mask_right_bound_p2] - self.world_box[3]
        mask_up_p1 = self.obs_all_raw[:,1:2] > self.mid_zone_x[0]
        mask_down_p1 = self.obs_all_raw[:,1:2] < self.lose_zone_x[0]
        self.obs_all_raw[:,1:2][mask_up_p1] = self.mid_zone_x[0]
        self.obs_all_raw[:,1:2][mask_down_p1] = self.lose_zone_x[0]
        mask_up_p2 = self.obs_all_raw[:,7:8] < self.mid_zone_x[1]
        mask_down_p2 = self.obs_all_raw[:,7:8] > self.lose_zone_x[1]
        self.obs_all_raw[:,7:8][mask_up_p2] = self.mid_zone_x[1]
        self.obs_all_raw[:,7:8][mask_down_p2] = self.lose_zone_x[1]

        mask_ball_right = self.obs_all_raw[:,12:13] >= (self.world_box[3] - self.ball_radius)
        mask_ball_left = self.obs_all_raw[:,12:13] <= (self.world_box[2] + self.ball_radius)
        self.obs_all_raw[:,12:13][mask_ball_left] -= 2*(self.obs_all_raw[:,12:13][mask_ball_left] - self.world_box[2] - self.ball_radius)
        self.obs_all_raw[:,12:13][mask_ball_right] -= 2*(self.obs_all_raw[:,12:13][mask_ball_right] - self.world_box[3] + self.ball_radius)
        self.obs_all_raw[:,14:15][np.logical_or(mask_ball_left, mask_ball_right)] *= -1



    def update_nxt_state(self, action):
        plank1_acc_line = action[:, :2]
        plank1_acc_angle = action[:,2:3]
        plank2_acc_line = -action[:, 3:5]
        plank2_acc_angle = action[:, 5:]

        plank1_acc_line = self.max_acc_line_plank * norm_batch(plank1_acc_line)
        plank1_acc_angle = self.max_acc_angle_plank * norm_batch(plank1_acc_angle)
        plank2_acc_line = self.max_acc_line_plank * norm_batch(plank2_acc_line)
        plank2_acc_angle = self.max_acc_angle_plank * norm_batch(plank2_acc_angle)

        plank1_vel = self.obs_all_raw[:, 3:5]
        plank1_heading = self.obs_all_raw[:, 5:6]
        plank2_vel = self.obs_all_raw[:, 9:11]
        plank2_heading = self.obs_all_raw[:, 11:12]

        ball_vel = self.obs_all_raw[:, 14:]

        self.obs_all_raw[:, :2] += plank1_vel * self.sim_dt
        self.obs_all_raw[:,2:3] += plank1_heading * self.sim_dt
        self.obs_all_raw[:,6:8] += plank2_vel * self.sim_dt
        self.obs_all_raw[:,8:9] += plank2_heading * self.sim_dt

        plank1_vel += plank1_acc_line * self.sim_dt
        plank1_heading += plank1_acc_angle * self.sim_dt
        plank2_vel += plank2_acc_line * self.sim_dt
        plank2_heading += plank2_acc_angle * self.sim_dt

        plank1_vel = self.max_vel_plank * norm_batch(plank1_vel / self.max_vel_plank)
        plank1_heading = self.max_heading_plank * norm_batch(plank1_heading / self.max_heading_plank)
        plank2_vel = self.max_vel_plank * norm_batch(plank2_vel / self.max_vel_plank)
        plank2_heading = self.max_heading_plank * norm_batch(plank2_heading / self.max_heading_plank)

        self.obs_all_raw[:, 3:5] = plank1_vel
        self.obs_all_raw[:, 5:6] = plank1_heading
        self.obs_all_raw[:, 9:11] = plank2_vel
        self.obs_all_raw[:, 11:12] = plank2_heading

        self.obs_all_raw[:, 12:14] += ball_vel *self.sim_dt
        self.obs_all_raw[:, 14:] = ball_vel

        self.constrain_terms()
        obs_raw_temp = copy.deepcopy(self.obs_all_raw)

        plank1_state = dict()
        plank1_state['position'] = obs_raw_temp[:, :2]
        plank1_state['orientation'] = obs_raw_temp[:, 2:3]
        plank1_state['velocity'] = obs_raw_temp[:, 3:5]
        plank1_state['bodyrate'] = obs_raw_temp[:, 5:6]
        plank1_state['plank_length'] = self.plank_length
        plank1_state['bounce_width'] = self.bounce_width + self.ball_radius

        ball_state = dict()
        ball_state['position'] = obs_raw_temp[:, 12:14]
        ball_state['velocity'] = obs_raw_temp[:, 14:]

        new_ball_vel = elastic_collision(plank1_state, ball_state)

        ball_state['velocity'] = new_ball_vel

        plank2_state = dict()
        plank2_state['position'] = obs_raw_temp[:, 6:8]
        plank2_state['orientation'] = obs_raw_temp[:, 8:9]
        plank2_state['velocity'] = obs_raw_temp[:, 9:11]
        plank2_state['bodyrate'] = obs_raw_temp[:, 11:12]
        plank2_state['plank_length'] = self.plank_length
        plank2_state['bounce_width'] = self.bounce_width + self.ball_radius

        new_ball_vel = elastic_collision(plank2_state, ball_state)

        ball_vel_norm = self.max_vel_ball * norm_batch(new_ball_vel/ self.max_vel_ball)

        self.obs_all_raw[:, 14:] = ball_vel_norm

    def get_reward_all(self):
        dones_all = np.zeros((self.num_threads, 2), dtype=bool)
        rewards_all = np.zeros((self.num_threads, 2), dtype=float)
        extra_info = [ dict() for i in range(self.num_threads) ]

        act_single_prev = copy.deepcopy(self.act_prev.reshape(self.num_threads, 2, 3))
        act_single = copy.deepcopy(self.act_record.reshape(self.num_threads, 2, 3))

        delta_act = np.sum(np.abs(act_single_prev - act_single), axis= -1)
        rewards_all -= self.delta_action_coef * delta_act

        rewards_all -= 1* self.step_coef
        ball_pos_y = self.obs_all_raw[:, 13]
        mask_player1_win = ball_pos_y > self.lose_zone_x[1]
        mask_player2_win = ball_pos_y < self.lose_zone_x[0]
        mask_timelimit = self.step_counter >= (self.episode_length -1)

        dones_all[np.logical_or(mask_player1_win, mask_player2_win),:] = True
        dones_all[mask_timelimit,:] = True

        discount = 1 - self.time_discount_coef * (self.step_counter + 1) / (self.episode_length)

        rewards_all[mask_player1_win, 0] += self.win_coef * discount[mask_player1_win]
        rewards_all[mask_player1_win, 1] -= self.lose_coef * discount[mask_player1_win]
        rewards_all[mask_player2_win, 1] += self.win_coef * discount[mask_player2_win]
        rewards_all[mask_player2_win, 0] -= self.lose_coef * discount[mask_player2_win]

        index_thread = np.arange(self.num_threads)

        for i in range(self.num_threads):
            for j in range(2):
                extra_info[i]["PLAYER_" + str(j) + "_step_penalty"] = - 1.0 *self.step_coef
                extra_info[i]["PLAYER_" + str(j) + "_delta_action_reward"] = -delta_act[i][j] * self.delta_action_coef

        for i in index_thread[mask_player1_win]:
            extra_info[i]["PLAYER_" + str(0) + "_win"] = 1.0
            extra_info[i]["PLAYER_" + str(0) + "_win_reward"] = 1.0 * self.win_coef * discount[i]
            extra_info[i]["PLAYER_" + str(1) + "_lose"] = 1.0
            extra_info[i]["PLAYER_" + str(1) + "_lose_reward"] = - 1.0 * self.lose_coef * discount[i]

        for i in index_thread[mask_player2_win]:
            extra_info[i]["PLAYER_" + str(1) + "_win"] = 1.0
            extra_info[i]["PLAYER_" + str(1) + "_win_reward"] = 1.0 * self.win_coef * discount[i]
            extra_info[i]["PLAYER_" + str(0) + "_lose"] = 1.0
            extra_info[i]["PLAYER_" + str(0) + "_lose_reward"] = - 1.0 * self.lose_coef * discount[i]

        for i in index_thread[mask_timelimit]:
            extra_info[i]["TimeLimit.truncated"] = 1.0
            extra_info[i]['PLAYER_1_draw'] = 1.0
            extra_info[i]['PLAYER_0_draw'] = 1.0

        for i in range(self.num_threads):
            for j in range(2):
                extra_info[i]["PLAYER_" + str(j) + "_total_reward"] = rewards_all[i][j]
                extra_info[i]["PLAYER_" + str(j) + "_total_done"] =  1.0 if dones_all[i][j] else 0.0

        return rewards_all, dones_all, extra_info
    
    def get_reward(self):
        rewards_all, dones_all, extra_info = self.get_reward_all()
        self.rewards = rewards_all[:,:1]
        self.dones = dones_all[:,:1]
        self.oppo_done = dones_all[:,1:]
        self.infos = extra_info

    def reset_all(self):
        player1_x = self.rng.uniform(self.world_box[2] + self.plank_length, self.world_box[3] - self.plank_length, self.num_threads)
        player2_x = self.rng.uniform(self.world_box[2] + self.plank_length, self.world_box[3] - self.plank_length, self.num_threads)
        player1_y = self.rng.uniform(self.lose_zone_x[0], self.mid_zone_x[0], self.num_threads)
        player2_y = self.rng.uniform(self.mid_zone_x[1], self.lose_zone_x[1], self.num_threads)
        ball_vel = self.rng.uniform(-1, 1, size=(self.num_threads, 2))
        ball_vel = self.init_vel_ball * norm_batch(ball_vel)
        player1_theta = self.rng.uniform(-np.pi/2, np.pi/2, self.num_threads)
        player2_theta = self.rng.uniform(-np.pi/2, np.pi/2, self.num_threads)

        self.ongoing[:] = False
        self.step_counter[:] = 0

        self.obs_all_raw = np.zeros_like(self.obs_all_raw)
        self.obs_all_raw[:, 0] = player1_x
        self.obs_all_raw[:, 1] = player1_y
        self.obs_all_raw[:, 2] = player1_theta
        self.obs_all_raw[:, 6] = player2_x
        self.obs_all_raw[:, 7] = player2_y
        self.obs_all_raw[:, 8] = player2_theta
        self.obs_all_raw[:, 14:] = ball_vel

    def reset_single(self, k):

        self.step_counter[k] = 0
        player1_x = self.rng.uniform(self.world_box[2] + self.plank_length, self.world_box[3] - self.plank_length, 1)
        player2_x = self.rng.uniform(self.world_box[2] + self.plank_length, self.world_box[3] - self.plank_length, 1)
        player1_y = self.rng.uniform(self.lose_zone_x[0], self.mid_zone_x[0], 1)
        player2_y = self.rng.uniform(self.mid_zone_x[1], self.lose_zone_x[1], 1)
        if self.rng.random() > 0.5:
            random_angle = self.rng.uniform(low=np.pi/4, high=3*np.pi/4)
        else:
            random_angle = self.rng.uniform(low=-3*np.pi/4, high=-np.pi/4)

        ball_vel = self.init_vel_ball * np.array([np.cos(random_angle), np.sin(random_angle)])
        player1_theta = self.rng.uniform(-np.pi/2, np.pi/2, 1)
        player2_theta = self.rng.uniform(-np.pi/2, np.pi/2, 1)

        obs_all_single = np.zeros(16)
        obs_all_single[0] = player1_x
        obs_all_single[1] = player1_y
        obs_all_single[2] = player1_theta
        obs_all_single[6] = player2_x
        obs_all_single[7] = player2_y
        obs_all_single[8] = player2_theta
        obs_all_single[14:] = ball_vel

        self.ongoing[k] = False

        self.obs_all_raw[k] = copy.deepcopy(obs_all_single)

    def reset(self):
        for i in range(self.num_threads):
            self.reset_single(i)
        self.gen_obs_from_raw()
        if self.oppo_policy is not None:
            self.oppo_rnn_states = np.zeros((self.num_threads, 1, self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, 1, 1), dtype=np.float32)
        return copy.deepcopy(self.obs)
    
    def step(self, act):
        action = self._reset_action(act)
        # action_p = defender_policy(copy.deepcopy(self.other_obs), copy.deepcopy(self.distances[:, 0, :]))
        self.act_prev[self.ongoing] = copy.deepcopy(self.act_record[self.ongoing])
        self.act_prev[self.ongoing == False] = copy.deepcopy(action[self.ongoing == False])
        self.act_record = copy.deepcopy(action)
        self.update_nxt_state(action)
        # obs = self.get_observation()
        self.get_reward()
        # print("sub_module_time = ", test_time_end - test_time_start)
        if self.oppo_policy is not None:
            self.oppo_rnn_states[self.oppo_done == True] = np.zeros(((self.oppo_done == True).sum(), self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, 1, 1), dtype=np.float32)
            self.oppo_masks[self.oppo_done == True] = np.zeros(((self.oppo_done == True).sum(), 1), dtype=np.float32)
        self.ongoing[self.ongoing == False] = True
        self.step_counter += 1
        self.isTerminal()
        self.gen_obs_from_raw()
        rewards_out  = copy.deepcopy(self.rewards[:, :, np.newaxis])
        obs_out = copy.deepcopy(self.obs)
        dones_out  = copy.deepcopy(self.dones)
        infos_out = copy.deepcopy(self.infos)
        
        return obs_out, rewards_out, dones_out, infos_out

    def isTerminal(self):
        for i in range(self.num_threads):
            if np.all(self.dones[i]):
                self.reset_single(i)

    def _reset_action(self, act):
        assert act.shape[1] == 1 * self.action_space[0].shape[0], 'wrong action dim!'
        # TODO: currently we randomly sample evader actions
        #evader_action = np.concatenate([np.expand_dims(np.concatenate([self.evader_action_space[i].sample() for i in range(self.num_evader_)]), axis=0) for _ in range(self.num_env_)], axis=0)
        if self.oppo_policy == None:
            oppo_action = np.concatenate([np.expand_dims(np.concatenate([self.oppo_act_space[i].sample() for i in range(1)]), axis=0) for _ in range(self.num_threads)], axis=0)
        else:
            oppo_action = self.get_oppo_action(self.oppo_obs)
        ## TODO: currently we clip the network action, in future ......
        #act = np.clip(act, -1, 1)
        #print(act, evader_action)
        action = np.concatenate([act, oppo_action], axis=1)
        return action

    @torch.no_grad()
    def get_oppo_action(self, oppo_obs):
        self.oppo_policy.actor.eval()
        oppo_action, oppo_rnn_states = self.oppo_policy.act(np.concatenate(oppo_obs),
                                                np.concatenate(self.oppo_rnn_states),
                                                np.concatenate(self.oppo_masks),
                                                deterministic=True)
        self.oppo_action = np.array(np.split(_t2n(oppo_action), self.num_threads))
        self.oppo_rnn_states = np.array(np.split(_t2n(oppo_rnn_states), self.num_threads))
        oppo_actions_env = np.concatenate([self.oppo_action[:, idx, :] for idx in range(1)], axis=1)

        return oppo_actions_env
    
def obs_2_dict(obs):
    plank1_state = dict()
    plank1_state['position'] = obs[:,:2]
    plank1_state['orientation'] = obs[:,2:3]

    plank2_state = dict()
    plank2_state['position'] = obs[:,6:8]
    plank2_state['orientation'] = obs[:,8:9]

    ball_state = dict()
    ball_state['position'] = obs[:,9:11]

    return plank1_state, plank2_state, ball_state


def parse_args(parser):
    parser.add_argument("--scenario_name", type=str,
                        default="simple_uav", 
                        help="which scenario to run on.")
    parser.add_argument("--num_agents", type=int, default=4,
                        help="number of controlled players.")
    parser.add_argument("--eval_deterministic", action="store_false", 
                        default=True, 
                        help="by default True. If False, sample action according to probability")
    parser.add_argument("--share_reward", action='store_false', 
                        default=True, 
                        help="by default true. If false, use different reward for each agent.")

    parser.add_argument("--save_videos", action="store_true", default=False, 
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_dir", type=str, default="", 
                        help="directory to save videos.")
    
    #added by junyu
    parser.add_argument("--encoder_layer_N",type=int,
                        default=1, help="number of encoder layers")
    parser.add_argument("--encoder_hidden_size", type=int,
                        default=32, help="hidden size of encoder")
    parser.add_argument("--proprio_shape", type=int, default=13,
                        help="proprio_shape")
    parser.add_argument("--teammate_shape", type=int, default=7,
                        help="teammate")
    parser.add_argument("--opponent_shape", type=int, default=3,
                        help="opponent_shape")
    parser.add_argument("--n_head", type=int, default=4, help="n_head")
    parser.add_argument("--d_k", type=int, default= 8, help="d_k")
    parser.add_argument("--attn_size", type=int, default=32, help="attn_size")

    return parser
    
        

if __name__ == "__main__":
    parser = get_config()
    parser = parse_args(parser)
    args=parser.parse_args()
    args.hidden_size = 128
    args.use_mixer = False
    envs = PlankAndBall(32)
    oppo_policy = Policy(args,
                        envs.oppo_obs_space[0],
                        envs.oppo_obs_space[0],
                        envs.oppo_act_space[0])
    envs.oppo_policy = Policy_P2B_straight(oppo_policy, envs.max_vel_plank, envs.max_heading_plank, envs.sim_dt, range(9,11), device = torch.device("cuda:0"))
    self_policy = Policy_P2B_rotate(oppo_policy, envs.max_vel_plank, envs.max_heading_plank, envs.sim_dt, range(9,11), device = torch.device("cuda:0"))
    obs = envs.reset()
    print(obs)
    obs_track = []
    oppo_obs_track = []
    dones_track = []
    obs_track.append(obs[:,0,:])
    oppo_obs_track.append(copy.deepcopy(envs.oppo_obs[:,0,:]))

    for i in range(400):
        # action_interface = np.concatenate([np.expand_dims(np.concatenate([envs.action_space[i].sample() for i in range(1)]), axis=0) for _ in range(envs.num_threads)], axis=0)
        action_interface = self_policy.action_generator.act(np.concatenate(envs.obs))
        obs, rewards, dones, infos = envs.step(action_interface)
        print("reward = ", rewards)
        obs_track.append(obs[:,0,:])
        oppo_obs_track.append(copy.deepcopy(envs.oppo_obs[:,0,:]))
        dones_track.append(dones)
        # print(obs[0])

    git_file_path = '/home/qiyuan/workspace/plot/20240321'

    obs_track = np.array(obs_track)
    oppo_obs_track = np.array(oppo_obs_track)
    dones_track = np.array(dones_track)
    # print(dones_track)

    env_state = dict()
    env_state['world_box'] = envs.world_box
    env_state['lose_zone_x'] = envs.lose_zone_x
    env_state['mid_zone_x'] = envs.mid_zone_x

    for i in range(8):

        for j in range(400):
            # print(dones_track[j,i])
            if dones_track[j,i].all():
                terminal_index = j
                break

        print("terminal_index = ", terminal_index)

        plank1_state, plank2_state, ball_state = obs_2_dict(obs_track[:terminal_index,i])

        plank1_state['plank_length'] = envs.plank_length
        plank2_state['plank_length'] = envs.plank_length
        plank1_state['plank_height'] = envs.bounce_width
        plank2_state['plank_height'] = envs.bounce_width
        ball_state['ball_radius'] = envs.ball_radius

        

        plot_gif_plankNball(git_file_path, 'test_obs', plank1_state, plank2_state, ball_state, env_state)

        print("{}/{} gif complete!".format(i,32))