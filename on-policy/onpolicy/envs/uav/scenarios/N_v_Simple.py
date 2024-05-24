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
import threading

from onpolicy.utils.plot3d_test import plot_track
from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_E2P_3Doptimal as evader_rule_policy
from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_P2E_straight as pursuer_rule_policy
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as multi_policy

def _t2n(x):
    return x.detach().cpu().numpy()

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
        self.close2lockdown_coef = self.reward_coef['close_to_lockdown_zone_coef']
        self.all_crash_coef = self.reward_coef['all_crash_coef']
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
    
    def get_reward_all(self):
        rewards_all = np.zeros((self.num_threads, self.total_num_agents))
        rewards_role = dict()
        extra_info = [ dict() for i in range(self.num_threads) ] 
        dones_all = np.zeros((self.num_threads, self.total_num_agents), dtype=bool)
        dones_role = dict()

        # dones_prev = self.dones_prev[k]
        distance_mat = copy.deepcopy(self.distances)
        distance_prev_mat = copy.deepcopy(self.distances_prev)

        mask = np.eye(self.total_num_agents, dtype=bool)
        distance_mat[:, mask] = np.inf
        distance_prev_mat[:, mask] = np.inf

        mask = ~self.active_state[:, np.newaxis, :]
        mask = np.repeat(mask, self.total_num_agents, axis=1)

        distance_mat[mask] = np.inf
        distance_prev_mat[mask] = np.inf

        
        obs_raw_single = copy.deepcopy(self.obs_all_raw.reshape(self.num_threads, self.total_num_agents, 3))
        obs_raw_prev = copy.deepcopy(self.obs_all_raw_prev.reshape(self.num_threads, self.total_num_agents, 3))
        act_single_prev = copy.deepcopy(self.act_prev.reshape(self.num_threads, self.total_num_agents, 4))
        act_single = copy.deepcopy(self.act_record.reshape(self.num_threads, self.total_num_agents, 4))

        mask_terrian = np.logical_and((obs_raw_single[:, :, 2] < self.min_height), self.active_state)
        mask_ceiling = np.logical_and((obs_raw_single[:, :, 2] > self.max_height), self.active_state)

        dis_pursuer = distance_mat[:, :self.num_role['pursuer'], :self.num_role['pursuer']]
        dis_p_e = distance_mat[:,self.total_num_agents -1, :self.num_role['pursuer']]
        dis_prev_p_e = distance_prev_mat[:,self.total_num_agents -1, :self.num_role['pursuer']]
        min_dis = np.min(dis_pursuer, axis= -1)
        pad_width = [(0, 0),  (0, 1)]
        mask_safe = np.pad(min_dis < self.min_dis_crash, pad_width=pad_width, mode='constant', constant_values=False)
        mask_safe = np.logical_and(mask_safe, self.active_state)

        dones_all[~self.active_state] = True

        # common rewards
        delta_act = np.sum(np.abs(act_single_prev - act_single), axis= -1)

        rewards_all[self.active_state] -= 1* self.step_coef
        rewards_all[self.active_state] -= delta_act[self.active_state] * self.delta_action_coef
        rewards_all[mask_terrian] -= 1 * self.dis2terrian_coef
        rewards_all[mask_ceiling] -= 1 * self.dis2ceiling_coef
        rewards_all[mask_safe] -= 1 * self.safe_dis_coef

        index_array = np.arange(self.total_num_agents)

        for i in range(self.num_threads):
            for j in index_array[self.active_state[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_step_penalty"] = - 1.0 *self.step_coef
                extra_info[i]["PLAYER_" + str(j) + "_delta_action_reward"] = -delta_act[i][j] * self.delta_action_coef

            for j in index_array[mask_terrian[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_approaching_terrian_reward"] = -1 * self.dis2terrian_coef
                extra_info[i]["PLAYER_" + str(j) + "_approaching_terrian_done"] = 1.0

            for j in index_array[mask_ceiling[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_approaching_ceiling_reward"] = -1 * self.dis2ceiling_coef
                extra_info[i]["PLAYER_" + str(j) + "_approaching_ceiling_done"] = 1.0

            for j in index_array[mask_safe[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_safe_distance_reward"] = -1 * self.safe_dis_coef
                extra_info[i]["PLAYER_" + str(j) + "_safe_distance_done"] = 1.0

        first_done = np.logical_or(np.logical_or(mask_terrian, mask_ceiling), mask_safe)

        dones_all[first_done] = True
        self.active_state[first_done] = False

        evader_alive = self.active_state[:, self.total_num_agents -1]
        evader_alive = evader_alive[:, np.newaxis]
        evader_alive = np.tile(evader_alive, (1, self.total_num_agents))
        # print(evader_alive)

        # pursuer rewards
        progress = np.zeros((self.num_threads, self.num_role['pursuer']), dtype=float)
        safe_index = self.active_state[:, :self.num_role['pursuer']]
        progress[safe_index] = dis_prev_p_e[safe_index] - dis_p_e[safe_index]
        progress = np.where(progress > 0, progress, 0)
        mask_progress = np.pad(progress>0, pad_width=pad_width, mode='constant', constant_values=False)
        ex_progress = np.pad(progress, pad_width=[(0,0), (0, 1)], mode='constant', constant_values=0)
        ex_progress[~evader_alive] = 0
        mask_progress = elementwise_and([mask_progress, self.active_state, evader_alive])
        # print(mask_progress)
        # mask_progress = mask_progress and self.active_state and evader_alive

        dis_xy = np.linalg.norm(obs_raw_single[:, self.total_num_agents -1, :2], axis=-1)
        dis_prev_xy = np.linalg.norm(obs_raw_prev[:, self.total_num_agents -1, :2], axis = -1)
        dis_xy = dis_xy[:, np.newaxis]
        dis_prev_xy = dis_prev_xy[:, np.newaxis]

        norm_dis_xy = (dis_xy - self.precaution_r) / (self.precaution_r - self.lockdown_r)

        close2Lock = np.tile(norm_dis_xy, (1, self.num_role['pursuer']))
        close2Lock = np.pad(close2Lock, pad_width=pad_width, mode='constant', constant_values=0)
        close2Lock[:, self.num_role['pursuer']: ] = -copy.deepcopy(norm_dis_xy)

        mask_lockdown = dis_xy < self.lockdown_r
        mask_loose_lock = np.tile(mask_lockdown, (1, self.num_role['pursuer']))
        mask_loose_lock = np.pad(mask_loose_lock, pad_width=pad_width, mode='constant', constant_values=False)
        mask_loose_lock = elementwise_and([mask_loose_lock, self.active_state])
        # mask_loose_lock = mask_loose_lock and self.active_state

        mask_enter_lock = np.pad(mask_lockdown, pad_width=[(0,0), (self.num_role['pursuer'], 0)], mode='constant', constant_values=False)
        mask_enter_lock = elementwise_and([mask_enter_lock, self.active_state])
        # mask_enter_lock = mask_enter_lock and self.active_state

        mask_out_range = dis_xy > self.precaution_r
        mask_out_range = np.pad(mask_out_range, pad_width=[(0,0), (self.num_role['pursuer'], 0)], mode='constant', constant_values=False)
        mask_out_range = elementwise_and([mask_out_range, self.active_state])
        # mask_out_range = mask_out_range and self.active_state

        min_p_e_dis = np.min(dis_p_e, axis = -1)
        min_p_e_dis = min_p_e_dis[:,np.newaxis]
        mask_seize = min_p_e_dis < self.min_dis_seize
        mask_seize_p = np.tile(mask_seize, (1, self.num_role['pursuer']))
        mask_seize_p = np.pad(mask_seize_p, pad_width=pad_width, mode='constant', constant_values=False)
        mask_seize_e = np.pad(mask_seize, pad_width=[(0,0), (self.num_role['pursuer'], 0)], mode='constant', constant_values=False)

        mask_seize_p = elementwise_and([mask_seize_p, self.active_state])
        mask_seize_e = elementwise_and([mask_seize_e, self.active_state])

        # mask_seize_p = mask_seize_p and self.active_state
        # mask_seize_e = mask_seize_e and self.active_state

        closeL = dis_prev_xy - dis_xy
        closeL = np.where(closeL > 0, closeL, 0)
        mask_closeL = np.pad(closeL>0, pad_width=[(0,0), (self.num_role['pursuer'], 0)], mode='constant', constant_values=False)
        mask_closeL = elementwise_and([mask_closeL, self.active_state])
        # mask_closeL = mask_closeL and self.active_state
        ex_closeL = np.pad(closeL, pad_width=[(0,0), (self.num_role['pursuer'], 0)], mode='constant', constant_values=0)

        second_done = elementwise_or([mask_loose_lock, mask_enter_lock, mask_out_range, mask_seize_p, mask_seize_e])
        # second_done = mask_loose_lock or mask_enter_lock or mask_out_range or mask_seize_p or mask_seize_e
        dones_all[second_done] = True
        self.active_state[second_done] = False

        rewards_all[mask_progress] += ex_progress[mask_progress] * self.pursuer_move2E_coef
        rewards_all[mask_loose_lock] -= 1 * self.pursuer_loseL_coef
        rewards_all[mask_enter_lock] += 1 * self.evader_enterL_coef
        rewards_all[mask_out_range] -= 1 * self.evader_out_preZone_coef
        rewards_all[mask_seize_e] -= 1*self.pursuer_seize_coef
        rewards_all[mask_seize_p] += 1*self.pursuer_seize_coef
        rewards_all[mask_closeL] += ex_closeL[mask_closeL] * self.evader_move2L_coef
        rewards_all[self.active_state] += close2Lock[self.active_state] * self.close2lockdown_coef

        for i in range(self.num_threads):
            for j in index_array[self.active_state[i]]:
                if j < self.num_role["pursuer"]:
                    extra_info[i]["PLAYER_" + str(j) + "_invasion_penalty"] = self.close2lockdown_coef * close2Lock[i][j]
                else:
                    extra_info[i]["PLAYER_" + str(j) + "_occupation_reward"] = self.close2lockdown_coef * close2Lock[i][j]

            for j in index_array[mask_progress[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_pursuer_move_to_evader_reward"] = ex_progress[i][j] * self.pursuer_move2E_coef

            for j in index_array[mask_loose_lock[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_pursuer_loose_lockdown_zone_reward"] = -1 * self.pursuer_loseL_coef
                extra_info[i]["PLAYER_" + str(j) + "_pursuer_loose_lockdown_zone_done"] = 1.0

            for j in index_array[mask_enter_lock[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_evader_enter_lockdown_zone_reward"] = 1 * self.evader_enterL_coef
                extra_info[i]["PLAYER_" + str(j) + "_evader_enter_lockdown_zone_done"] = 1.0

            for j in index_array[mask_out_range[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_evader_out_of_range_reward"] = -1 * self.evader_out_preZone_coef
                extra_info[i]["PLAYER_" + str(j) + "_evader_out_of_range_done"] = 1.0

            for j in index_array[mask_seize_e[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_pursuer_seize_evader_penalty"] = -1*self.pursuer_seize_coef
                extra_info[i]["PLAYER_" + str(j) + "_evader_seized_done"] = 1.0

            for j in index_array[mask_seize_p[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_pursuer_seize_evader_reward"] = 1*self.pursuer_seize_coef
                extra_info[i]["PLAYER_" + str(j) + "_pursuer_seize_evader_done"] = 1.0

            for j in index_array[mask_closeL[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_evader_move_to_lockdown_zone_reward"] = ex_closeL[i][j] * self.evader_move2L_coef
        
        pursuer_done = np.all(dones_all[:, :self.num_role['pursuer']], axis=-1)
        evader_done = dones_all[:, self.total_num_agents -1]
        mask_timelimit = self.step_counter >= (self.episode_length -1)

        mask_crash_lose = np.logical_xor(pursuer_done, evader_done)
        ex_mask_crash_lose = np.tile(mask_crash_lose[:,np.newaxis], (1, self.total_num_agents))

        mask_all_crash = np.logical_and(ex_mask_crash_lose, ~dones_all)

        rewards_all[mask_all_crash] += 1.0 * self.all_crash_coef

        for i in range(self.num_threads):
            for j in index_array[mask_all_crash[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_opponent_all_crash_reward"] = 1.0 * self.all_crash_coef
                extra_info[i]["PLAYER_" + str(j) + "_opponent_all_crash_done"] = 1.0

        terminal_done = elementwise_or([pursuer_done, evader_done, mask_timelimit])

        # terminal_done = pursuer_done or evader_done or mask_timelimit

        dones_all[terminal_done, :] = True
        self.active_state[terminal_done, :] = False



        for i in range(self.num_threads):
            for j in range(self.total_num_agents):
                extra_info[i]["PLAYER_" + str(j) + "_total_reward"] = rewards_all[i][j]
                extra_info[i]["PLAYER_" + str(j) + "_total_done"] =  1.0 if dones_all[i][j] else 0.0
            
        index_thread_array = np.arange(self.num_threads)
        for i in index_thread_array[mask_timelimit]:
            extra_info[i]["TimeLimit.truncated"] = 1.0

        rewards_role['pursuer'] = rewards_all[:, :self.num_role['pursuer']]
        rewards_role['evader'] = rewards_all[:, self.num_role['pursuer']:]

        dones_role['pursuer'] = dones_all[:, :self.num_role['pursuer']]
        dones_role['evader'] = dones_all[:, self.num_role['pursuer']:]

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
            self.reset_single(i)
        self.gen_obs_from_raw()
        if self.oppo_policy is not None:
            self.oppo_rnn_states = np.zeros((self.num_threads, self.num_oppo, self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, self.num_oppo, 1), dtype=np.float32)
        return copy.deepcopy(self.obs)       

    def get_reward(self):
        rewards_role, dones_role, info = self.get_reward_all()
        self.rewards = rewards_role[self.team_name]
        self.dones = dones_role[self.team_name]
        self.oppo_done = dones_role[self.oppo_name]
        self.infos = info
        # self.infos = []
        # for i in range(self.num_threads):
        #     rewards_role, dones_role, info = self.get_reward_single(i)
        #     self.rewards[i] = rewards_role[self.team_name]
        #     self.dones[i] = dones_role[self.team_name]
        #     self.oppo_done[i] = dones_role[self.oppo_name]
        #     self.infos.append(info)


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

        # if np.isnan(vel_array).any() or np.isinf(vel_array).any():
        #     print("vel_array has inf/nan!")
        #     print("vel_array = ", vel_array)
        # print("vel = ", vel_array)
        self.obs_all_raw_prev = copy.deepcopy(self.obs_all_raw)
        if mask is not None:
            vel_array *= mask
        self.obs_all_raw += vel_array * self.sim_dt

    def step(self, act):
        test_time_start = time.time()
        action = self._reset_action(act)
        action_mask = np.zeros((self.num_threads, 3*self.total_num_agents))
        extend_active_state = np.repeat(self.active_state, 3, axis=1)
        action_mask[extend_active_state] = 1.0
        action_mask[~extend_active_state] = 0.0
        self.act_prev[self.ongoing] = copy.deepcopy(self.act_record[self.ongoing])
        self.act_prev[self.ongoing == False] = copy.deepcopy(action[self.ongoing == False])
        self.act_record = copy.deepcopy(action)
        self.update_obs_raw(action, action_mask)
        # obs = self.get_observation()
        self.get_reward()
        # print("sub_module_time = ", test_time_end - test_time_start)
        if self.oppo_policy is not None:
            self.oppo_rnn_states[self.oppo_done == True] = np.zeros(((self.oppo_done == True).sum(), self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, self.num_oppo, 1), dtype=np.float32)
            self.oppo_masks[self.oppo_done == True] = np.zeros(((self.oppo_done == True).sum(), 1), dtype=np.float32)
        self.ongoing[self.ongoing == False] = True
        self.step_counter += 1
        self.isTerminal()
        self.gen_obs_from_raw()
        rewards_out  = copy.deepcopy(self.rewards[:, :, np.newaxis])
        obs_out = copy.deepcopy(self.obs)
        dones_out  = copy.deepcopy(self.dones)
        infos_out = copy.deepcopy(self.infos)

        # if np.isnan(obs_out).any() or np.isinf(obs_out).any():
        #     print("obs_out has inf/nan!")
        #     print("obs_out = ", obs_out)

        # if np.isnan(rewards_out).any() or np.isinf(rewards_out).any():
        #     print("rewards_out has inf/nan!")
        #     print("rewards_out = ", rewards_out)
    
        # if not np.isin(dones_out, [True, False]).all():
        #     print("dones wrong! dones = ", dones_out)

        test_time_end = time.time()
        self.sub_time += test_time_end - test_time_start
        
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
    n_th = 32
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
    policies.append(evader_rule_policy(oppo_policy, 7, path, grid , device=torch.device("cuda:0")))
    # policies.append(pursuer_rule_policy(oppo_policy, 7, 9, device=torch.device("cpu")))
    envs.oppo_policy = multi_policy(1, policies)
    obs = envs.reset()
    print("reset obs = ", obs)
    ob_op = copy.deepcopy(envs.oppo_obs)

    team_track = []
    oppo_track = []


    team_track.append([obs[0][i][0:3] for i in range(envs.num_team)])
    oppo_track.append([ob_op[0][i][0:3] for i in range(envs.num_oppo)])

    # print("sub_role_size = ", envs.sub_role_shape)

    game_long = 2000

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
        ob_op = copy.deepcopy(envs.oppo_obs)
        oppo_track.append([ob_op[0][i][0:3] for i in range(envs.num_oppo)])
        #print("Step:", i, nxt[0][0][0:3], don, rew)
        # print("\nStep:", i)
        # print("info = ", inf)
        # print("reward = ", rew)
        # print("obs=", nxt[0])
        # print("done=", don)
        # print(ob_op[0][0][0:3], nxt[0][0][0:3])
        # print(nxt[0][1][0:3])
        # print(nxt[0][2][0:3])
        # print("\n")

    end_time = time.time()

    print("fps = ", (game_long * n_th)/(end_time - start_time))
    print("total time = ", end_time - start_time)
    print("sub_time = ", envs.sub_time)

    #print(len(oppo_track[0]))
    #print(team_track[1][0], team_track[1][0][0])
    # print(envs.num_team, envs.num_oppo)
    # print(oppo_track)
    # plot_track(oppo_track, team_track)