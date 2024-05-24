import numpy as np
from onpolicy.envs.uav.scenarios.vec_env_wrapper import SimpleBase
from onpolicy.envs.uav.scenario import BaseScenarioUAV
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
from gym import spaces
from onpolicy.config import get_config
import yaml
import copy
import time
import torch
from multiprocessing import Pool

from onpolicy.utils.plot3d_test import plot_track
from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_E2P_3Doptimal as evader_rule_policy
from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_P2E_straight as pursuer_rule_policy
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as multi_policy

def _t2n(x):
    return x.detach().cpu().numpy()


class NvSimple_single:
    def __init__(self, team_name, oppo_name, seed, cfg_path = None):
        self.team_name = team_name
        self.oppo_name = oppo_name
        self.rng = np.random.default_rng(seed)
        if cfg_path is None:
            cfg_path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/envs/uav/scenarios/simple_config.yaml'
        self.cfg_ = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

        self.config = self.cfg_['quadrotor_env']
        self.role_keys = self.cfg_['role']

        self.episode_length = self.config['episode_length']
        self.step_counter = 0
        self.ongoing = False

        roles = [1 if r == 'pursuer' else 0 for r in self.cfg_['role']]
        num_pursuer = sum(roles)
        num_evader = len(roles) - num_pursuer
        self.total_num_agents = len(roles)
        self.num_role = dict()
        self.num_role['pursuer'] = num_pursuer
        self.num_role['evader'] = num_evader

        self.num_team = self.num_role[self.team_name]
        self.num_oppo = self.num_role[self.oppo_name]

        # agent configs
        self.agent_cfg = self.config['quadrotor']
        self.sim_dt = self.agent_cfg['sim_dt']
        self.max_vel_p = self.agent_cfg['max_vel_pursuer']
        self.max_vel_e = self.agent_cfg['max_vel_evader']
        self.max_heading_p = self.agent_cfg['max_heading_rate_pursuer']
        self.max_heading_e = self.agent_cfg['max_heading_rate_evader']

        # env configs
        self.lockdown_r = self.config['env']['lockdown_zone_radius']
        self.control_r = self.config['env']['control_zone_radius']
        self.precaution_r = self.config['env']['precaution_zone_radius']
        self.lower_height = self.config['env']['lower_height']
        self.upper_height = self.config['env']['upper_height']

        # terminal configs
        self.min_height = self.config['terminal_conditions']['min_height']
        self.max_height = self.config['terminal_conditions']['max_height']
        self.min_dis_crash = self.config['terminal_conditions']['min_safe_distance']
        self.min_dis_seize = self.config['terminal_conditions']['min_pursuer_seize_evader_distance']
        self.max_delta_action = self.config['terminal_conditions']['max_delta_action']

        # reward configs
        self.reward_coef = self.config['reward']
        self.delta_action_coef = self.reward_coef['delta_action_coef']
        self.safe_dis_coef = self.reward_coef['safe_distance_coef']
        self.dis2terrian_coef = self.reward_coef['distance_to_terrian_coef']
        self.dis2ceiling_coef = self.reward_coef['distance_to_ceiling_coef']
        self.pursuer_move2E_coef = self.reward_coef['pursuer_move_to_evader_coef']
        self.pursuer_loseL_coef = self.reward_coef['pursuer_loose_lockdown_zone_coef']
        self.pursuer_seize_coef = self.reward_coef['pursuer_seize_evader_coef']
        self.evader_move2L_coef = self.reward_coef['evader_move_to_lockdown_zone_coef']
        self.evader_enterL_coef = self.reward_coef['evader_enter_lockdown_zone_coef']
        self.evader_out_preZone_coef = self.reward_coef['evader_out_of_precaution_zone_coef']
        self.step_coef = self.reward_coef['step_coef']

        self.obs_role = dict()
        self.obs_role['pursuer'] = np.zeros((self.num_role['pursuer'], 3*self.total_num_agents))
        self.obs_role['evader'] = np.zeros((self.num_role['evader'], 3*self.total_num_agents))
        self.obs_all_raw = np.zeros((3*self.total_num_agents))
        self.obs_all_raw_prev = np.zeros((3*self.total_num_agents))
        self.rewards = np.zeros((self.num_role[self.team_name]))
        self.dones = np.zeros((self.num_role[self.team_name]), dtype=bool)
        self.oppo_done = np.zeros((self.num_role[self.oppo_name]), dtype=bool)
        self.dones_prev = np.zeros((self.num_role[self.team_name]), dtype=bool)
        self.act_record = np.zeros((4*self.total_num_agents))
        self.act_prev = np.zeros((4*self.total_num_agents))
        self.distances = np.zeros((self.total_num_agents, self.total_num_agents))
        self.distances_prev = np.zeros((self.total_num_agents, self.total_num_agents))
        self.active_state = np.ones((self.total_num_agents), dtype=bool)

    def reset(self):

        self.step_counter = 0
        height = self.rng.uniform(self.lower_height, self.upper_height, self.total_num_agents)
        degree = self.rng.uniform(0, 2 * np.pi, self.total_num_agents)

        radius_p = self.rng.uniform(0, self.control_r, self.num_role['pursuer'])
        radius_e = self.rng.uniform(self.control_r, self.precaution_r, self.num_role['evader'])
        radius = np.concatenate([radius_p, radius_e])

        obs_all_single = np.zeros(3*self.total_num_agents)

        self.ongoing = False

        for i in range(self.total_num_agents):
            obs_all_single[3*i] = radius[i] * np.cos(degree[i])
            obs_all_single[3*i+1] = radius[i] * np.sin(degree[i])
            obs_all_single[3*i+2] = height[i]

        self.active_state[:] = True

        self.obs_all_raw = copy.deepcopy(obs_all_single)
        self.obs_all_raw_prev = copy.deepcopy(obs_all_single)

        return obs_all_single

    def get_reward(self):
        rewards_all = np.zeros(self.total_num_agents)
        rewards_role = dict()
        extra_info = dict()
        dones_all = np.zeros(self.total_num_agents, dtype=bool)
        dones_role = dict()

        # dones_prev = self.dones_prev[k]
        distance_mat = copy.deepcopy(self.distances)
        distance_prev_mat = copy.deepcopy(self.distances_prev)
        for i in range(self.total_num_agents):
            distance_mat[i][i] = np.inf
            distance_prev_mat[i][i] = np.inf
            distance_mat[i][self.active_state == False] = np.inf
            distance_prev_mat[i][self.active_state == False] = np.inf
        # distance_mat[distance_mat < 1e-8] = np.inf
        # distance_prev_mat[distance_prev_mat < 1e-8] = np.inf
        obs_raw_single = copy.deepcopy(self.obs_all_raw.reshape(self.total_num_agents, 3))
        obs_raw_prev = copy.deepcopy(self.obs_all_raw_prev.reshape(self.total_num_agents, 3))
        act_single_prev = copy.deepcopy(self.act_prev.reshape(self.total_num_agents, 4))
        act_single = copy.deepcopy(self.act_record.reshape(self.total_num_agents, 4))

        # common rewards
        delta_act = np.sum(np.abs(act_single_prev - act_single), axis= -1)
        for i in range(self.total_num_agents):
            if self.active_state[i]:
                rewards_all[i] -= 1* self.step_coef
                extra_info["PLAYER_" + str(i) + "_step_penalty"] = - 1.0 *self.step_coef
                rewards_all[i] -= delta_act[i] * self.delta_action_coef
                extra_info["PLAYER_" + str(i) + "_delta_action_reward"] = -delta_act[i] * self.delta_action_coef

                if obs_raw_single[i][2] < self.min_height:
                    dones_all[i] = True
                    self.active_state[i] = False
                    rewards_all[i] -= 1 * self.dis2terrian_coef
                    extra_info["PLAYER_" + str(i) + "_approaching_terrian_reward"] = -1 * self.dis2terrian_coef
                    extra_info["PLAYER_" + str(i) + "_approaching_terrian_done"] = 1.0

                if obs_raw_single[i][2] > self.max_height:
                    dones_all[i] = True
                    self.active_state[i] = False
                    rewards_all[i] -= 1 * self.dis2ceiling_coef
                    extra_info["PLAYER_" + str(i) + "_approaching_ceiling_reward"] = -1 * self.dis2ceiling_coef
                    extra_info["PLAYER_" + str(i) + "_approaching_ceiling_done"] = 1.0

                if i < self.num_role['pursuer']:
                    min_dis = np.min(distance_mat[i][:self.num_role['pursuer']])
                    if min_dis < self.min_dis_crash:
                        rewards_all[i] -= 1 * self.safe_dis_coef
                        dones_all[i] = True
                        self.active_state[i] = False
                        extra_info["PLAYER_" + str(i) + "_safe_distance_reward"] = -1 * self.safe_dis_coef
                        extra_info["PLAYER_" + str(i) + "_safe_distance_done"] = 1.0
            else:
                dones_all[i] = True

        # pursuer rewards
        for i in range(self.num_role['pursuer']):
            if self.active_state[i]:
                if self.active_state[self.total_num_agents - 1]:
                    progress = distance_prev_mat[i][self.total_num_agents - 1] - distance_mat[i][self.total_num_agents -1]
                else:
                    progress = 0
                
                if progress > 0:
                    rewards_all[i] += progress * self.pursuer_move2E_coef
                    extra_info["PLAYER_" + str(i) + "_pursuer_move_to_evader_reward"] = progress * self.pursuer_move2E_coef

                if np.linalg.norm(obs_raw_single[self.total_num_agents -1][:2]) < self.lockdown_r:
                    rewards_all[i] -= 1 * self.pursuer_loseL_coef
                    dones_all[i] = True
                    self.active_state[i] = False
                    extra_info["PLAYER_" + str(i) + "_pursuer_loose_lockdown_zone_reward"] = -1 * self.pursuer_loseL_coef
                    extra_info["PLAYER_" + str(i) + "_pursuer_loose_lockdown_zone_done"] = 1.0

                if np.min(distance_mat[self.total_num_agents -1]) < self.min_dis_seize:
                    rewards_all[i] += 1*self.pursuer_seize_coef
                    dones_all[i] = True
                    self.active_state[i] = False
                    extra_info["PLAYER_" + str(i) + "_pursuer_seize_evader_reward"] = 1*self.pursuer_seize_coef
                    extra_info["PLAYER_" + str(i) + "_pursuer_seize_evader_done"] = 1.0
            else:
                dones_all[i] = True

        # evader rewards
        for i in range(self.num_role['pursuer'], self.total_num_agents):
            if self.active_state[i]:
                progress = np.linalg.norm(obs_raw_prev[i][:2]) - np.linalg.norm(obs_raw_single[i][:2])
                if progress > 0:
                    rewards_all[i] += progress * self.evader_move2L_coef
                    extra_info["PLAYER_" + str(i) + "_evader_move_to_lockdown_zone_reward"] = progress * self.evader_move2L_coef

                if np.linalg.norm(obs_raw_single[i][:2]) < self.lockdown_r:
                    rewards_all[i] += 1 * self.evader_enterL_coef
                    dones_all[i] = True
                    self.active_state[i] = False
                    extra_info["PLAYER_" + str(i) + "_evader_enter_lockdown_zone_reward"] = 1 * self.evader_enterL_coef
                    extra_info["PLAYER_" + str(i) + "_evader_enter_lockdown_zone_done"] = 1.0

                # add seizure penalty
                if np.min(distance_mat[i]) < self.min_dis_seize:
                    rewards_all[i] -= 1*self.pursuer_seize_coef
                    dones_all[i] = True
                    self.active_state[i] = False
                    extra_info["PLAYER_" + str(i) + "_pursuer_seize_evader_penalty"] = -1*self.pursuer_seize_coef
                    extra_info["PLAYER_" + str(i) + "_evader_seized_done"] = 1.0

                if np.linalg.norm(obs_raw_single[i][:2]) > self.precaution_r:
                    rewards_all[i] -= 1 * self.evader_out_preZone_coef
                    dones_all[i] = True
                    self.active_state[i] = False
                    extra_info["PLAYER_" + str(i) + "_evader_out_of_range_reward"] = -1 * self.evader_out_preZone_coef
                    extra_info["PLAYER_" + str(i) + "_evader_out_of_range_done"] = 1.0
            else:
                dones_all[i] = True

        if sum(dones_all) == self.total_num_agents - 1 and dones_all[self.total_num_agents -1] == False:
            dones_all[self.total_num_agents -1] = True
            self.active_state[self.total_num_agents -1] = False

        if dones_all[self.total_num_agents -1]:
            for i in range(self.total_num_agents - 1):
                dones_all[i] = True
                self.active_state[i] = False

        for i in range(self.total_num_agents):
            extra_info["PLAYER_" + str(i) + "_total_reward"] = rewards_all[i]
            extra_info["PLAYER_" + str(i) + "_total_done"] =  1.0 if dones_all[i] else 0.0

        # for i in range(self.total_num_agents):
        #     if dones_prev[i] == False and dones_all[i] == True and sum(dones_all) < self.total_num_agents:
        #         extra_info["PLAYER_" + str(i) + "_early_dead"] = 1.0

        if self.step_counter >= self.episode_length -1:
            extra_info["TimeLimit.truncated"] = 1.0
            # self.reset_single(k)
            for i in range(self.total_num_agents):
                dones_all[i] = True
                self.active_state[i] = False

        rewards_role['pursuer'] = rewards_all[:self.num_role['pursuer']]
        rewards_role['evader'] = rewards_all[self.num_role['pursuer']:]

        dones_role['pursuer'] = dones_all[:self.num_role['pursuer']]
        dones_role['evader'] = dones_all[self.num_role['pursuer']:]

        return rewards_role, dones_role, extra_info
    
    def isTerminal(self):
        if np.all(self.dones):
            self.reset()
    
    def gen_obs_from_raw(self):
        if self.ongoing:
            self.distances_prev = copy.deepcopy(self.distances)
        reshaped_obs = copy.deepcopy(self.obs_all_raw.reshape(self.total_num_agents, 3))
        diff = reshaped_obs[ :, np.newaxis, :] - reshaped_obs[ np.newaxis, :, :]
        self.distances = np.linalg.norm(diff, axis=2)
        if self.ongoing == False:
            self.distances_prev = copy.deepcopy(self.distances)

        for j in range(self.total_num_agents):
            agent_pos = copy.deepcopy(self.obs_all_raw[3*j:3*j+3])
            other_pos = np.concatenate([copy.deepcopy(self.obs_all_raw[:3*j]), copy.deepcopy(self.obs_all_raw[3*j+3:])], axis=0)
            con_obs = np.concatenate([agent_pos, other_pos], axis=0)
            reshape_con_obs = con_obs.reshape(self.total_num_agents, 3)
            if j < self.num_role['pursuer']:
                teammates_distances = np.concatenate([self.distances[j, 0:j], self.distances[j, j+1:self.num_role['pursuer']]], axis= -1)
                opponents_distances = self.distances[j, self.num_role['pursuer']:]

                sorted_teammates_indices = np.argsort(teammates_distances, axis=0) + 1
                sorted_opponents_indices = np.argsort(opponents_distances, axis=0) + self.num_role['pursuer']

                sorted_indices = np.hstack((
                    np.zeros((1), dtype=int),
                    sorted_teammates_indices,
                    sorted_opponents_indices
                ))
                
                sorted_re_obs = reshape_con_obs[sorted_indices]
                self.obs_role['pursuer'][j, :] = sorted_re_obs.reshape((3*self.total_num_agents))
            else:
                teammates_distances = np.concatenate([self.distances[j, self.num_role['pursuer']:j], self.distances[j, j+1:]], axis= -1)
                opponents_distances = self.distances[j, :self.num_role['pursuer']]

                sorted_teammates_indices = np.argsort(teammates_distances, axis=0) + 1
                sorted_opponents_indices = np.argsort(opponents_distances, axis=0) + self.num_role['evader']

                sorted_indices = np.hstack((
                    np.zeros((1), dtype=int),
                    sorted_teammates_indices,
                    sorted_opponents_indices
                ))
            
                sorted_re_obs = reshape_con_obs[sorted_indices]
                self.obs_role['evader'][j - self.num_role['pursuer'], :] = sorted_re_obs.reshape((3*self.total_num_agents))

        self.obs = copy.deepcopy(self.obs_role[self.team_name])
        self.oppo_obs = copy.deepcopy(self.obs_role[self.oppo_name])

    def update_obs_raw(self, action, mask = None):
        vel_array = np.zeros((3*self.total_num_agents))
        for i in range(self.total_num_agents):
            action_norm = copy.deepcopy(action[4*i:4*i+3])
            norms = np.linalg.norm(action_norm, axis=0)
            if norms > 1:
                action_norm /= norms

            if i < self.num_role['pursuer']:
                vel_array[3*i:3*i+3] = self.max_vel_p * action_norm
            else:
                vel_array[3*i:3*i+3] = self.max_vel_e * action_norm

        # if np.isnan(vel_array).any() or np.isinf(vel_array).any():
        #     print("vel_array has inf/nan!")
        #     print("vel_array = ", vel_array)
        # print("vel = ", vel_array)
        self.obs_all_raw_prev = copy.deepcopy(self.obs_all_raw)
        if mask is not None:
            vel_array *= mask
        self.obs_all_raw += vel_array * self.sim_dt

    def step(self, action):
        # action = self._reset_action(act)
        action_mask = np.zeros((3*self.total_num_agents))
        extend_active_state = np.repeat(self.active_state, 3, axis=0)
        action_mask[extend_active_state] = 1.0
        action_mask[~extend_active_state] = 0.0
        self.act_prev[self.ongoing] = copy.deepcopy(self.act_record[self.ongoing])
        self.act_prev[self.ongoing == False] = copy.deepcopy(action[self.ongoing == False])
        self.act_record = copy.deepcopy(action)
        self.update_obs_raw(action, action_mask)
        # obs = self.get_observation()
        rewards_role, dones_role, extra_info  = self.get_reward()
        self.rewards = rewards_role[self.team_name]
        self.dones = dones_role[self.team_name]
        self.infos = extra_info
        # print("sub_module_time = ", test_time_end - test_time_start)
        self.ongoing = True
        self.step_counter += 1
        self.isTerminal()
        self.gen_obs_from_raw()
        rewards_out  = copy.deepcopy(self.rewards)
        obs_out = copy.deepcopy(self.obs)
        dones_out  = copy.deepcopy(self.dones)
        infos_out = copy.deepcopy(self.infos)

        return obs_out, rewards_out, dones_out, infos_out
    


def step_worker(args):
    Nv_single, action = args
    observation, reward, done, info = Nv_single.step(action)
    return observation, reward, done, info



class NvSimple:
    def __init__(self, num_threads, team_name, oppo_name, cfg_path = None, oppo_policy=None):
        self.team_name = team_name
        self.oppo_name = oppo_name
        self.num_threads = num_threads

        if cfg_path is None:
            cfg_path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/envs/uav/scenarios/simple_config.yaml'
        self.cfg_ = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

        self.config = self.cfg_['quadrotor_env']
        self.role_keys = self.cfg_['role']

        self.episode_length = self.config['episode_length']
        self.step_counter = np.zeros(num_threads)
        self.ongoing = np.zeros(num_threads, dtype=bool)

        roles = [1 if r == 'pursuer' else 0 for r in self.cfg_['role']]
        num_pursuer = sum(roles)
        num_evader = len(roles) - num_pursuer
        self.total_num_agents = len(roles)
        self.num_role = dict()
        self.num_role['pursuer'] = num_pursuer
        self.num_role['evader'] = num_evader

        self.num_team = self.num_role[self.team_name]
        self.num_oppo = self.num_role[self.oppo_name]

        # agent configs
        self.agent_cfg = self.config['quadrotor']
        self.sim_dt = self.agent_cfg['sim_dt']
        self.max_vel_p = self.agent_cfg['max_vel_pursuer']
        self.max_vel_e = self.agent_cfg['max_vel_evader']
        self.max_heading_p = self.agent_cfg['max_heading_rate_pursuer']
        self.max_heading_e = self.agent_cfg['max_heading_rate_evader']

        # env configs
        self.lockdown_r = self.config['env']['lockdown_zone_radius']
        self.control_r = self.config['env']['control_zone_radius']
        self.precaution_r = self.config['env']['precaution_zone_radius']
        self.lower_height = self.config['env']['lower_height']
        self.upper_height = self.config['env']['upper_height']

        # terminal configs
        self.min_height = self.config['terminal_conditions']['min_height']
        self.max_height = self.config['terminal_conditions']['max_height']
        self.min_dis_crash = self.config['terminal_conditions']['min_safe_distance']
        self.min_dis_seize = self.config['terminal_conditions']['min_pursuer_seize_evader_distance']
        self.max_delta_action = self.config['terminal_conditions']['max_delta_action']

        # reward configs
        self.reward_coef = self.config['reward']
        self.delta_action_coef = self.reward_coef['delta_action_coef']
        self.safe_dis_coef = self.reward_coef['safe_distance_coef']
        self.dis2terrian_coef = self.reward_coef['distance_to_terrian_coef']
        self.dis2ceiling_coef = self.reward_coef['distance_to_ceiling_coef']
        self.pursuer_move2E_coef = self.reward_coef['pursuer_move_to_evader_coef']
        self.pursuer_loseL_coef = self.reward_coef['pursuer_loose_lockdown_zone_coef']
        self.pursuer_seize_coef = self.reward_coef['pursuer_seize_evader_coef']
        self.evader_move2L_coef = self.reward_coef['evader_move_to_lockdown_zone_coef']
        self.evader_enterL_coef = self.reward_coef['evader_enter_lockdown_zone_coef']
        self.evader_out_preZone_coef = self.reward_coef['evader_out_of_precaution_zone_coef']
        self.step_coef = self.reward_coef['step_coef']

        # set spaces   
        self.observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(3*self.total_num_agents,), dtype=np.float32) for _ in range(self.num_role[self.team_name])]
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32) for _ in range(self.num_role[self.team_name])]

        self.oppo_obs_space = [spaces.Box(low=-1.0, high=1.0, shape=(3*self.total_num_agents,), dtype=np.float32) for _ in range(self.num_role[self.oppo_name])]
        self.oppo_act_space = [spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32) for _ in range(self.num_role[self.oppo_name])]


        self.oppo_policy = copy.deepcopy(oppo_policy)
        self.obs_role = dict()
        self.obs_role['pursuer'] = np.zeros((self.num_threads, self.num_role['pursuer'], 3*self.total_num_agents))
        self.obs_role['evader'] = np.zeros((self.num_threads, self.num_role['evader'], 3*self.total_num_agents))
        # self.obs = np.zeros((self.num_threads, self.num_role[self.team_name], 3*self.total_num_agents))
        # self.oppo_obs = np.zeros((self.num_threads, self.num_role[self.oppo_name], 3*self.total_num_agents))
        self.obs_all_raw = np.zeros((self.num_threads, 3*self.total_num_agents))
        self.obs_all_raw_prev = np.zeros((self.num_threads, 3*self.total_num_agents))
        self.rewards = np.zeros((self.num_threads, self.num_role[self.team_name]))
        self.dones = np.zeros((self.num_threads, self.num_role[self.team_name]), dtype=bool)
        self.oppo_done = np.zeros((self.num_threads, self.num_role[self.oppo_name]), dtype=bool)
        self.dones_prev = np.zeros((self.num_threads, self.num_role[self.team_name]), dtype=bool)
        self.act_record = np.zeros((self.num_threads, 4*self.total_num_agents))
        self.act_prev = np.zeros((self.num_threads, 4*self.total_num_agents))
        self.distances = np.zeros((self.num_threads, self.total_num_agents, self.total_num_agents))
        self.distances_prev = np.zeros((self.num_threads, self.total_num_agents, self.total_num_agents))
        self.active_state = np.ones((self.num_threads, self.total_num_agents), dtype=bool)

        seed = int((int(time.time()*100) % 10000) * 1000)
        self.rng = np.random.default_rng(seed)
        self.Nv_agents = []

        for _ in range(self.num_threads):
            self.Nv_agents.append(NvSimple_single(self.team_name, self.oppo_name, self.rng.integers(low=0, high=2**32 - 1), cfg_path=cfg_path))
    
    def _reset_action(self, act):
        assert act.shape[1] == self.num_team * self.action_space[0].shape[0], 'wrong action dim!'
        # TODO: currently we randomly sample evader actions
        #evader_action = np.concatenate([np.expand_dims(np.concatenate([self.evader_action_space[i].sample() for i in range(self.num_evader_)]), axis=0) for _ in range(self.num_env_)], axis=0)
        if self.oppo_policy == None:
            oppo_action = np.concatenate([np.expand_dims(np.concatenate([self.oppo_act_space[i].sample() for i in range(self.num_role[self.oppo_name])]), axis=0) for _ in range(self.num_threads)], axis=0)
        else:
            oppo_action = self.get_oppo_action(self.oppo_obs)
        ## TODO: currently we clip the network action, in future ......
        #act = np.clip(act, -1, 1)
        #print(act, evader_action)
        if self.team_name in self.role_keys[0]:
            action = np.concatenate([act, oppo_action], axis=1)
        else:
            action = np.concatenate([oppo_action, act], axis=1)
        return action

    def get_observation(self):
        self.gen_obs_from_raw()
        return copy.deepcopy(self.obs)
    
    def gen_obs_from_raw(self):
        self.distances_prev[self.ongoing] = copy.deepcopy(self.distances[self.ongoing])
        reshaped_obs = copy.deepcopy(self.obs_all_raw.reshape(self.num_threads, self.total_num_agents, 3))
        diff = reshaped_obs[:, :, np.newaxis, :] - reshaped_obs[:, np.newaxis, :, :]
        self.distances = np.linalg.norm(diff, axis=3)
        self.distances_prev[self.ongoing == False] = copy.deepcopy(self.distances[self.ongoing == False])

        for j in range(self.total_num_agents):
            agent_pos = copy.deepcopy(self.obs_all_raw[:, 3*j:3*j+3])
            other_pos = np.concatenate([copy.deepcopy(self.obs_all_raw[:, :3*j]), copy.deepcopy(self.obs_all_raw[:, 3*j+3:])], axis=1)
            con_obs = np.concatenate([agent_pos, other_pos], axis=1)
            reshape_con_obs = con_obs.reshape(self.num_threads, self.total_num_agents, 3)
            if j < self.num_role['pursuer']:
                teammates_distances = np.concatenate([self.distances[:, j, 0:j], self.distances[:, j, j+1:self.num_role['pursuer']]], axis= -1)
                opponents_distances = self.distances[:, j, self.num_role['pursuer']:]

                sorted_teammates_indices = np.argsort(teammates_distances, axis=1) + 1
                sorted_opponents_indices = np.argsort(opponents_distances, axis=1) + self.num_role['pursuer']

                sorted_indices = np.hstack((
                    np.zeros((self.num_threads, 1), dtype=int),
                    sorted_teammates_indices,
                    sorted_opponents_indices
                ))
                for k in range(self.num_threads):
                    sorted_re_obs = reshape_con_obs[k][sorted_indices[k]]
                    self.obs_role['pursuer'][k, j, :] = sorted_re_obs.reshape((1, 3*self.total_num_agents))
            else:
                teammates_distances = np.concatenate([self.distances[:, j, self.num_role['pursuer']:j], self.distances[:, j, j+1:]], axis= -1)
                opponents_distances = self.distances[:, j, :self.num_role['pursuer']]

                sorted_teammates_indices = np.argsort(teammates_distances, axis=1) + 1
                sorted_opponents_indices = np.argsort(opponents_distances, axis=1) + self.num_role['evader']

                sorted_indices = np.hstack((
                    np.zeros((self.num_threads, 1), dtype=int),
                    sorted_teammates_indices,
                    sorted_opponents_indices
                ))
                for k in range(self.num_threads):
                    sorted_re_obs = reshape_con_obs[k][sorted_indices[k]]
                    self.obs_role['evader'][k, j - self.num_role['pursuer'], :] = sorted_re_obs.reshape((1, 3*self.total_num_agents))

        self.obs = copy.deepcopy(self.obs_role[self.team_name])
        self.oppo_obs = copy.deepcopy(self.obs_role[self.oppo_name])
    
    def get_reward_single(self, k):
        rewards_all = np.zeros(self.total_num_agents)
        rewards_role = dict()
        extra_info = dict()
        dones_all = np.zeros(self.total_num_agents, dtype=bool)
        dones_role = dict()

        # dones_prev = self.dones_prev[k]
        distance_mat = copy.deepcopy(self.distances[k])
        distance_prev_mat = copy.deepcopy(self.distances_prev[k])
        for i in range(self.total_num_agents):
            distance_mat[i][i] = np.inf
            distance_prev_mat[i][i] = np.inf
            distance_mat[i][self.active_state[k] == False] = np.inf
            distance_prev_mat[i][self.active_state[k] == False] = np.inf
        # distance_mat[distance_mat < 1e-8] = np.inf
        # distance_prev_mat[distance_prev_mat < 1e-8] = np.inf
        obs_raw_single = copy.deepcopy(self.obs_all_raw[k].reshape(self.total_num_agents, 3))
        obs_raw_prev = copy.deepcopy(self.obs_all_raw_prev[k].reshape(self.total_num_agents, 3))
        act_single_prev = copy.deepcopy(self.act_prev[k].reshape(self.total_num_agents, 4))
        act_single = copy.deepcopy(self.act_record[k].reshape(self.total_num_agents, 4))

        # common rewards
        delta_act = np.sum(np.abs(act_single_prev - act_single), axis= -1)
        for i in range(self.total_num_agents):
            if self.active_state[k][i]:
                rewards_all[i] -= 1* self.step_coef
                extra_info["PLAYER_" + str(i) + "_step_penalty"] = - 1.0 *self.step_coef
                rewards_all[i] -= delta_act[i] * self.delta_action_coef
                extra_info["PLAYER_" + str(i) + "_delta_action_reward"] = -delta_act[i] * self.delta_action_coef

                if obs_raw_single[i][2] < self.min_height:
                    dones_all[i] = True
                    self.active_state[k][i] = False
                    rewards_all[i] -= 1 * self.dis2terrian_coef
                    extra_info["PLAYER_" + str(i) + "_approaching_terrian_reward"] = -1 * self.dis2terrian_coef
                    extra_info["PLAYER_" + str(i) + "_approaching_terrian_done"] = 1.0

                if obs_raw_single[i][2] > self.max_height:
                    dones_all[i] = True
                    self.active_state[k][i] = False
                    rewards_all[i] -= 1 * self.dis2ceiling_coef
                    extra_info["PLAYER_" + str(i) + "_approaching_ceiling_reward"] = -1 * self.dis2ceiling_coef
                    extra_info["PLAYER_" + str(i) + "_approaching_ceiling_done"] = 1.0

                if i < self.num_role['pursuer']:
                    min_dis = np.min(distance_mat[i][:self.num_role['pursuer']])
                    if min_dis < self.min_dis_crash:
                        rewards_all[i] -= 1 * self.safe_dis_coef
                        dones_all[i] = True
                        self.active_state[k][i] = False
                        extra_info["PLAYER_" + str(i) + "_safe_distance_reward"] = -1 * self.safe_dis_coef
                        extra_info["PLAYER_" + str(i) + "_safe_distance_done"] = 1.0
            else:
                dones_all[i] = True

        # pursuer rewards
        for i in range(self.num_role['pursuer']):
            if self.active_state[k][i]:
                if self.active_state[k][self.total_num_agents - 1]:
                    progress = distance_prev_mat[i][self.total_num_agents - 1] - distance_mat[i][self.total_num_agents -1]
                else:
                    progress = 0
                
                if progress > 0:
                    rewards_all[i] += progress * self.pursuer_move2E_coef
                    extra_info["PLAYER_" + str(i) + "_pursuer_move_to_evader_reward"] = progress * self.pursuer_move2E_coef

                if np.linalg.norm(obs_raw_single[self.total_num_agents -1][:2]) < self.lockdown_r:
                    rewards_all[i] -= 1 * self.pursuer_loseL_coef
                    dones_all[i] = True
                    self.active_state[k][i] = False
                    extra_info["PLAYER_" + str(i) + "_pursuer_loose_lockdown_zone_reward"] = -1 * self.pursuer_loseL_coef
                    extra_info["PLAYER_" + str(i) + "_pursuer_loose_lockdown_zone_done"] = 1.0

                if np.min(distance_mat[self.total_num_agents -1]) < self.min_dis_seize:
                    rewards_all[i] += 1*self.pursuer_seize_coef
                    dones_all[i] = True
                    self.active_state[k][i] = False
                    extra_info["PLAYER_" + str(i) + "_pursuer_seize_evader_reward"] = 1*self.pursuer_seize_coef
                    extra_info["PLAYER_" + str(i) + "_pursuer_seize_evader_done"] = 1.0
            else:
                dones_all[i] = True

        # evader rewards
        for i in range(self.num_role['pursuer'], self.total_num_agents):
            if self.active_state[k][i]:
                progress = np.linalg.norm(obs_raw_prev[i][:2]) - np.linalg.norm(obs_raw_single[i][:2])
                if progress > 0:
                    rewards_all[i] += progress * self.evader_move2L_coef
                    extra_info["PLAYER_" + str(i) + "_evader_move_to_lockdown_zone_reward"] = progress * self.evader_move2L_coef

                if np.linalg.norm(obs_raw_single[i][:2]) < self.lockdown_r:
                    rewards_all[i] += 1 * self.evader_enterL_coef
                    dones_all[i] = True
                    self.active_state[k][i] = False
                    extra_info["PLAYER_" + str(i) + "_evader_enter_lockdown_zone_reward"] = 1 * self.evader_enterL_coef
                    extra_info["PLAYER_" + str(i) + "_evader_enter_lockdown_zone_done"] = 1.0

                # add seizure penalty
                if np.min(distance_mat[i]) < self.min_dis_seize:
                    rewards_all[i] -= 1*self.pursuer_seize_coef
                    dones_all[i] = True
                    self.active_state[k][i] = False
                    extra_info["PLAYER_" + str(i) + "_pursuer_seize_evader_penalty"] = -1*self.pursuer_seize_coef
                    extra_info["PLAYER_" + str(i) + "_evader_seized_done"] = 1.0

                if np.linalg.norm(obs_raw_single[i][:2]) > self.precaution_r:
                    rewards_all[i] -= 1 * self.evader_out_preZone_coef
                    dones_all[i] = True
                    self.active_state[k][i] = False
                    extra_info["PLAYER_" + str(i) + "_evader_out_of_range_reward"] = -1 * self.evader_out_preZone_coef
                    extra_info["PLAYER_" + str(i) + "_evader_out_of_range_done"] = 1.0
            else:
                dones_all[i] = True

        if sum(dones_all) == self.total_num_agents - 1 and dones_all[self.total_num_agents -1] == False:
            dones_all[self.total_num_agents -1] = True
            self.active_state[k][self.total_num_agents -1] = False

        if dones_all[self.total_num_agents -1]:
            for i in range(self.total_num_agents - 1):
                dones_all[i] = True
                self.active_state[k][i] = False

        for i in range(self.total_num_agents):
            extra_info["PLAYER_" + str(i) + "_total_reward"] = rewards_all[i]
            extra_info["PLAYER_" + str(i) + "_total_done"] =  1.0 if dones_all[i] else 0.0

        # for i in range(self.total_num_agents):
        #     if dones_prev[i] == False and dones_all[i] == True and sum(dones_all) < self.total_num_agents:
        #         extra_info["PLAYER_" + str(i) + "_early_dead"] = 1.0

        if self.step_counter[k] >= self.episode_length -1:
            extra_info["TimeLimit.truncated"] = 1.0
            # self.reset_single(k)
            for i in range(self.total_num_agents):
                dones_all[i] = True
                self.active_state[k][i] = False

        rewards_role['pursuer'] = rewards_all[:self.num_role['pursuer']]
        rewards_role['evader'] = rewards_all[self.num_role['pursuer']:]

        dones_role['pursuer'] = dones_all[:self.num_role['pursuer']]
        dones_role['evader'] = dones_all[self.num_role['pursuer']:]

        return rewards_role, dones_role, extra_info
    
    def reset_single(self, k):

        self.step_counter[k] = 0
        height = self.rng.uniform(self.lower_height, self.upper_height, self.total_num_agents)
        degree = self.rng.uniform(0, 2 * np.pi, self.total_num_agents)

        radius_p = self.rng.uniform(0, self.control_r, self.num_role['pursuer'])
        radius_e = self.rng.uniform(self.control_r, self.precaution_r, self.num_role['evader'])
        radius = np.concatenate([radius_p, radius_e])

        obs_all_single = np.zeros(3*self.total_num_agents)

        self.ongoing[k] = False

        for i in range(self.total_num_agents):
            obs_all_single[3*i] = radius[i] * np.cos(degree[i])
            obs_all_single[3*i+1] = radius[i] * np.sin(degree[i])
            obs_all_single[3*i+2] = height[i]

        self.active_state[k, :] = True

        self.obs_all_raw[k] = copy.deepcopy(obs_all_single)
        self.obs_all_raw_prev[k] = copy.deepcopy(obs_all_single)

    def reset(self):
        self.sub_time = 0

        for i in range(self.num_threads):
            self.obs_all_raw[i] = self.Nv_agents[i].reset()
        
        self.obs_all_raw_prev = copy.deepcopy(self.obs_all_raw)
        self.gen_obs_from_raw()
        if self.oppo_policy is not None:
            self.oppo_rnn_states = np.zeros((self.num_threads, self.num_oppo, self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, self.num_oppo, 1), dtype=np.float32)
        return copy.deepcopy(self.obs)       

    def get_reward(self):
        self.infos = []
        for i in range(self.num_threads):
            rewards_role, dones_role, info = self.get_reward_single(i)
            self.rewards[i] = rewards_role[self.team_name]
            self.dones[i] = dones_role[self.team_name]
            self.oppo_done[i] = dones_role[self.oppo_name]
            self.infos.append(info)

    def update_obs_raw(self, action, mask = None):
        vel_array = np.zeros((self.num_threads, 3*self.total_num_agents))
        for i in range(self.total_num_agents):
            action_norm = copy.deepcopy(action[:,4*i:4*i+3])
            norms = np.linalg.norm(action_norm, axis=1)
            large_norms = norms > 1
            action_norm[large_norms] /= norms[large_norms, np.newaxis]

            if i < self.num_role['pursuer']:
                vel_array[:, 3*i:3*i+3] = self.max_vel_p * action_norm
            else:
                vel_array[:, 3*i:3*i+3] = self.max_vel_e * action_norm

        if np.isnan(vel_array).any() or np.isinf(vel_array).any():
            print("vel_array has inf/nan!")
            print("vel_array = ", vel_array)
        # print("vel = ", vel_array)
        self.obs_all_raw_prev = copy.deepcopy(self.obs_all_raw)
        if mask is not None:
            vel_array *= mask
        self.obs_all_raw += vel_array * self.sim_dt

    def step(self, act):
        action = self._reset_action(act)
        agent_action_pairs = [(agent, action[i]) for i, agent in enumerate(self.Nv_agents)]
        obs_temp = []
        rewards_temp = [] 
        dones_temp = []
        info_temp = []

        for i in range(self.num_threads):
            obs_temp_s, rewards_temp_s, dones_temp_s, info_temp_s = step_worker(agent_action_pairs[i])
            obs_temp.append(obs_temp_s)
            rewards_temp.append(rewards_temp_s)
            dones_temp.append(dones_temp_s)
            info_temp.append(info_temp_s)


        # with Pool(processes = self.num_threads) as pool:
        #     results = pool.map(step_worker, agent_action_pairs)

        # obs_temp, rewards_temp, dones_temp, info_temp = zip(*results)
        
        rewards_out  = copy.deepcopy(np.array(rewards_temp)[:, :, np.newaxis])
        obs_out = copy.deepcopy(np.array(obs_temp))
        dones_out  = copy.deepcopy(np.array(dones_temp))
        infos_out = copy.deepcopy(info_temp)
        
        return obs_out, rewards_out, dones_out, infos_out

    def isTerminal(self):
        for i in range(self.num_threads):
            if np.all(self.dones[i]):
                self.reset_single(i)

    
    @torch.no_grad()
    def get_oppo_action(self, oppo_obs):
        self.oppo_policy.actor.eval()
        oppo_action, oppo_rnn_states = self.oppo_policy.act(np.concatenate(oppo_obs),
                                                np.concatenate(self.oppo_rnn_states),
                                                np.concatenate(self.oppo_masks),
                                                deterministic=True)
        self.oppo_action = np.array(np.split(_t2n(oppo_action), self.num_threads))
        self.oppo_rnn_states = np.array(np.split(_t2n(oppo_rnn_states), self.num_threads))
        oppo_actions_env = np.concatenate([self.oppo_action[:, idx, :] for idx in range(self.num_oppo)], axis=1)

        return oppo_actions_env

    

def restore(oppo_policy, model_dir, use_mixer = True):
    """Restore policy's networks from a saved model."""
    policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
    oppo_policy.actor.load_state_dict(policy_actor_state_dict)
    if use_mixer:
        policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer.pt')
        oppo_policy.mixer.load_state_dict(policy_mixer_state_dict)

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
    n_th = 4
    envs=NvSimple(n_th, 'pursuer', 'evader')
    model_dir= "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/scripts/train_uav_scripts/wandb/run-20231109_011724-3ilxzbv9/files"
    oppo_policy = Policy(args,
                            envs.oppo_obs_space[0],
                            envs.oppo_obs_space[0],
                            envs.oppo_act_space[0])
    
    #restore(pursuer_policy, model_dir)
    print(args.hidden_size)
    #envs.oppo_policy = copy.deepcopy(pursuer_policy)
    #envs.oppo_policy = evader_rule_policy(oppo_policy, 3, 0.57/2)
    # path = "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattacker.mat"
    # samples = np.linspace(-105, 105, 51)
    path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattackerV3.mat'
    grid = (np.linspace(25, 205, 46), np.linspace(-100, 100, 51), np.linspace(0, 100, 26),
            np.linspace(-5, 205, 36), np.linspace(-5, 205, 36))
    policies = []
    policies.append(evader_rule_policy(oppo_policy, 7, path, grid , device=torch.device("cpu")))
    # policies.append(pursuer_rule_policy(oppo_policy, 7, 9, device=torch.device("cpu")))
    # envs.oppo_policy = multi_policy(1, policies)
    obs = envs.reset()
    print("reset obs = ", obs)
    ob_op = copy.deepcopy(envs.oppo_obs)

    team_track = []
    oppo_track = []


    team_track.append([obs[0][i][0:3] for i in range(envs.num_team)])
    oppo_track.append([ob_op[0][i][0:3] for i in range(envs.num_oppo)])

    # print("sub_role_size = ", envs.sub_role_shape)

    game_long = 10

    # for i in range(10):
    #     obs = envs.reset()
    #     print(obs)

    start_time = time.time()
    for i in range(game_long):
        # action_interface = [np.array([0.5,0.5,0,0]), np.array([0.9,-0.2,0,0]), np.array([-0.5,0.5,0,0])]
        #action_interface = [np.array([0,0,0.5,0])]
        action_interface = np.concatenate([np.expand_dims(np.concatenate([envs.action_space[i].sample() for i in range(envs.num_team)]), axis=0) for _ in range(envs.num_threads)], axis=0)
        # action_interface = np.expand_dims(np.concatenate(action_interface), axis=0)
        # print("act = ", action_interface[0])
        nxt, rew, don, inf = envs.step(action_interface)
        # if don[0].all():
        #     break
        team_track.append([nxt[0][i][0:3] for i in range(envs.num_team)])
        # print("team_track = ", team_track)
        # ob_op = copy.deepcopy(envs.oppo_obs)
        # oppo_track.append([ob_op[0][i][0:3] for i in range(envs.num_oppo)])
        #print("Step:", i, nxt[0][0][0:3], don, rew)
        print("\nStep:", i)
        # print("info = ", inf)
        # print("reward = ", rew)
        print("obs=", nxt[0])
        # print("done=", don)
        # print(ob_op[0][0][0:3], nxt[0][0][0:3])
        # print(nxt[0][1][0:3])
        # print(nxt[0][2][0:3])
        print("\n")

    end_time = time.time()

    print("fps = ", (game_long * n_th)/(end_time - start_time))
    print("total time = ", end_time - start_time)
    print("sub_time = ", envs.sub_time)

    #print(len(oppo_track[0]))
    #print(team_track[1][0], team_track[1][0][0])
    # print(envs.num_team, envs.num_oppo)
    # print(oppo_track)
    # plot_track(oppo_track, team_track)