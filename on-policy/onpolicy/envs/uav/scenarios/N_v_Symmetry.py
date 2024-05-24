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

def defender_target(obs, defender_r, distance_mat):
    n_th = distance_mat.shape[0]
    distance_mat[:, 0] = np.inf
    min_dis_inx = np.argmin(distance_mat, axis=1)
    closest_pos = np.zeros((n_th, 3))
    for i in range(n_th):
        closest_pos[i] = obs[i, 0, 3*min_dis_inx[i]:3*min_dis_inx[i]+3]

    norm_xy = np.linalg.norm(closest_pos[:,:2], axis= -1, keepdims=True)
    normalized_pos = defender_r * closest_pos[:,:2] / norm_xy
    target_pos = np.concatenate([normalized_pos, closest_pos[:,2:]], axis= -1)
    return target_pos

def defender_policy(obs, distance_mat):
    n_th = distance_mat.shape[0]
    distance_mat[:, 0] = np.inf
    min_dis_inx = np.argmin(distance_mat, axis=1)
    closest_pos = np.zeros((n_th, 3))
    min_dis = np.zeros(n_th)
    for i in range(n_th):
        closest_pos[i] = obs[i, 0, 3*min_dis_inx[i]:3*min_dis_inx[i]+3]
        min_dis[i] = distance_mat[i, min_dis_inx[i]]

    self_pos = obs[:, 0, :3]
    delta_pos = closest_pos - self_pos
    action_p = np.zeros((n_th, 4))
    norm_xy = np.linalg.norm(delta_pos, axis= -1, keepdims=True) + 1e-4
    action_p[:,:3] = delta_pos / norm_xy
    
    return action_p

def defender_policy_new(obs):
    n_th = obs.shape[0]
    oppo_obs = obs[:, 0, 3:].reshape(n_th,2,3)
    distance_mat = np.linalg.norm(oppo_obs[:,:,:2], axis= - 1)
    min_dis_inx = np.argmin(distance_mat, axis=1) + 1
    closest_pos = np.zeros((n_th, 3))
    for i in range(n_th):
        closest_pos[i] = obs[i, 0, 3*min_dis_inx[i]:3*min_dis_inx[i]+3]

    self_pos = obs[:, 0, :3]
    delta_pos = closest_pos - self_pos
    action_p = np.zeros((n_th, 4))
    norm_xy = np.linalg.norm(delta_pos, axis= -1, keepdims=True) + 1e-4
    action_p[:,:3] = delta_pos / norm_xy
    
    return action_p



class NvSymmetry:
    def __init__(self, num_threads, cfg_path = None, oppo_policy=None):
        self.num_threads = num_threads

        if cfg_path is None:
            cfg_path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/envs/uav/scenarios/symmetry_config.yaml'
        self.cfg_ = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

        self.config = self.cfg_['quadrotor_env']
        self.role_keys = self.cfg_['role']

        self.episode_length = self.config['episode_length']
        self.step_counter = np.zeros(num_threads)
        self.ongoing = np.zeros(num_threads, dtype=bool)

        roles = [1 if r == 'pursuer' else 0 for r in self.cfg_['role']]
        self.total_num_agents = len(roles)

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
        self.dis2terrian_coef = self.reward_coef['distance_to_terrian_coef']
        self.dis2ceiling_coef = self.reward_coef['distance_to_ceiling_coef']
        self.pursuer_move2E_coef = self.reward_coef['pursuer_move_to_evader_coef']
        self.evader_move2L_coef = self.reward_coef['evader_move_to_lockdown_zone_coef']
        self.win_coef = self.reward_coef['win_coef']
        self.lose_coef = self.reward_coef['lose_coef']
        self.step_coef = self.reward_coef['step_coef']

        # set spaces   
        self.observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(3*self.total_num_agents,), dtype=np.float32) for _ in range(1)]
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32) for _ in range(1)]

        self.oppo_obs_space = [spaces.Box(low=-1.0, high=1.0, shape=(3*self.total_num_agents,), dtype=np.float32) for _ in range(1)]
        self.oppo_act_space = [spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32) for _ in range(1)]


        self.oppo_policy = copy.deepcopy(oppo_policy)
        # self.obs = np.zeros((self.num_threads, self.num_role[self.team_name], 3*self.total_num_agents))
        # self.oppo_obs = np.zeros((self.num_threads, self.num_role[self.oppo_name], 3*self.total_num_agents))
        self.obs_all_raw = np.zeros((self.num_threads, 3*self.total_num_agents))
        self.obs_all_raw_prev = np.zeros((self.num_threads, 3*self.total_num_agents))
        self.rewards = np.zeros((self.num_threads, 1))
        self.dones = np.zeros((self.num_threads, 1), dtype=bool)
        self.oppo_done = np.zeros((self.num_threads, 1), dtype=bool)
        self.dones_prev = np.zeros((self.num_threads, 1), dtype=bool)
        self.act_record = np.zeros((self.num_threads, 4*3))
        self.act_prev = np.zeros((self.num_threads, 4*3))
        self.distances = np.zeros((self.num_threads, self.total_num_agents, self.total_num_agents))
        self.distances_prev = np.zeros((self.num_threads, self.total_num_agents, self.total_num_agents))
        self.active_state = np.ones((self.num_threads, 3), dtype=bool)

        seed = int((int(time.time()*100) % 10000) * 1000)
        self.rng = np.random.default_rng(seed)
    
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

    
    def gen_obs_from_raw(self):
        self.distances_prev[self.ongoing] = copy.deepcopy(self.distances[self.ongoing])
        reshaped_obs = copy.deepcopy(self.obs_all_raw.reshape(self.num_threads, self.total_num_agents, 3))
        diff = reshaped_obs[:, :, np.newaxis, :] - reshaped_obs[:, np.newaxis, :, :]
        self.distances = np.linalg.norm(diff, axis=3)
        self.distances_prev[self.ongoing == False] = copy.deepcopy(self.distances[self.ongoing == False])

        self.other_obs = copy.deepcopy(self.obs_all_raw).reshape(self.num_threads, 1, 3*self.total_num_agents)

        agent_pos = copy.deepcopy(self.obs_all_raw[:, 3:6])
        other_pos = np.concatenate([copy.deepcopy(self.obs_all_raw[:, 6:]), copy.deepcopy(self.obs_all_raw[:, :3])], axis=1)
        self.obs = np.concatenate([agent_pos, other_pos], axis=1).reshape(self.num_threads, 1, 3*self.total_num_agents)

        agent_pos = copy.deepcopy(self.obs_all_raw[:, 6:9])
        other_pos = np.concatenate([copy.deepcopy(self.obs_all_raw[:, 3:6]), copy.deepcopy(self.obs_all_raw[:, 0:3])], axis=1)
        self.oppo_obs = np.concatenate([agent_pos, other_pos], axis=1).reshape(self.num_threads, 1, 3*self.total_num_agents)
    
    
    def get_reward_all(self):
        rewards_all = np.zeros((self.num_threads, 2))
        rewards_role = dict()
        extra_info = [ dict() for i in range(self.num_threads) ] 
        dones_all = np.zeros((self.num_threads, 2), dtype=bool)
        dones_role = dict()
        
        obs_raw_single = copy.deepcopy(self.obs_all_raw.reshape(self.num_threads, self.total_num_agents, 3))
        obs_raw_prev = copy.deepcopy(self.obs_all_raw_prev.reshape(self.num_threads, self.total_num_agents, 3))
        act_single_prev = copy.deepcopy(self.act_prev[:, 4:].reshape(self.num_threads, 2, 4))
        act_single = copy.deepcopy(self.act_record[:, 4:].reshape(self.num_threads, 2, 4))

        mask_terrian = (obs_raw_single[:, 1:, 2] < self.min_height)
        mask_ceiling = (obs_raw_single[:, 1:, 2] > self.max_height)

        distance_mat = copy.deepcopy(self.distances)
        distance_prev_mat = copy.deepcopy(self.distances_prev)

        pad_width = [(0, 0),  (0, 1)]

        # common rewards
        delta_act = np.sum(np.abs(act_single_prev - act_single), axis= -1)

        index_array = np.arange(2)

        dis_xy = np.linalg.norm(obs_raw_single[:, 1:, :2], axis=-1)
        dis_prev_xy = np.linalg.norm(obs_raw_prev[:, 1:, :2], axis = -1)
        mask_lockdown = dis_xy < self.lockdown_r
        mask_out_range = dis_xy > self.precaution_r

        dis_p_e = distance_mat[:,1:, 0]
        mask_seize = dis_p_e < self.min_dis_seize

        closeL = dis_prev_xy - dis_xy
        closeL = np.where(closeL > 0, closeL, 0)
        mask_closeL = closeL > 0

        lose_done = elementwise_or([mask_terrian, mask_ceiling, mask_out_range, mask_seize])

        exchange_lose_done = np.concatenate([lose_done[:,1:], lose_done[:,:1]], axis=-1)

        win_done = mask_lockdown

        exchange_win_done = np.concatenate([win_done[:,1:], win_done[:,:1]], axis=-1)

        lose_done = elementwise_or([lose_done, exchange_win_done])
        win_done = elementwise_or([win_done, exchange_lose_done])

        bi_lose_done = elementwise_and([lose_done[:,0], lose_done[:,1]])
        bi_win_done = elementwise_and([win_done[:,0], win_done[:,1]])
        draw_done = elementwise_or([bi_win_done, bi_lose_done])
        draw_done = np.tile(draw_done[:,np.newaxis], (1, 2))

        win_done[draw_done] = False
        lose_done[draw_done] = False

        # win_done

        dones_all = elementwise_or([win_done, lose_done, draw_done])

        # second_done = mask_loose_lock or mask_enter_lock or mask_out_range or mask_seize_p or mask_seize_e

        rewards_all -= 1* self.step_coef
        rewards_all -= delta_act * self.delta_action_coef

        rewards_all[mask_closeL] += closeL[mask_closeL] * self.evader_move2L_coef
        rewards_all[win_done] += 1*self.win_coef
        rewards_all[lose_done] -= 1*self.lose_coef

        for i in range(self.num_threads):
            for j in index_array:
                extra_info[i]["PLAYER_" + str(j) + "_step_penalty"] = - 1.0 *self.step_coef
                extra_info[i]["PLAYER_" + str(j) + "_delta_action_reward"] = -delta_act[i][j] * self.delta_action_coef

            for j in index_array[mask_terrian[i]]:
                # extra_info[i]["PLAYER_" + str(j) + "_approaching_terrian_reward"] = -1 * self.dis2terrian_coef
                extra_info[i]["PLAYER_" + str(j) + "_approaching_terrian_done"] = 1.0

            for j in index_array[mask_ceiling[i]]:
                # extra_info[i]["PLAYER_" + str(j) + "_approaching_ceiling_reward"] = -1 * self.dis2ceiling_coef
                extra_info[i]["PLAYER_" + str(j) + "_approaching_ceiling_done"] = 1.0

            for j in index_array[mask_lockdown[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_enter_lockdown_zone_done"] = 1.0

            for j in index_array[mask_out_range[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_out_of_range_done"] = 1.0

            for j in index_array[mask_seize[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_seized_done"] = 1.0

            for j in index_array[mask_closeL[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_evader_move_to_lockdown_zone_reward"] = closeL[i][j] * self.evader_move2L_coef

            for j in index_array[win_done[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_win"] = 1.0
                extra_info[i]["PLAYER_" + str(j) + "_win_reward"] = 1.0 * self.win_coef

            for j in index_array[lose_done[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_lose"] = 1.0
                extra_info[i]["PLAYER_" + str(j) + "_lose_reward"] = - 1.0 * self.lose_coef

            for j in index_array[draw_done[i]]:
                extra_info[i]["PLAYER_" + str(j) + "_draw"] = 1.0


        
        self_done = dones_all[:, 0]
        oppo_done = dones_all[:, 1]
        mask_timelimit = self.step_counter >= (self.episode_length -1)

        terminal_done = elementwise_or([self_done, oppo_done, mask_timelimit])

        # terminal_done = pursuer_done or evader_done or mask_timelimit

        dones_all[terminal_done, :] = True

        for i in range(self.num_threads):
            for j in range(2):
                extra_info[i]["PLAYER_" + str(j) + "_total_reward"] = rewards_all[i][j]
                extra_info[i]["PLAYER_" + str(j) + "_total_done"] =  1.0 if dones_all[i][j] else 0.0
            
        index_thread_array = np.arange(self.num_threads)
        for i in index_thread_array[mask_timelimit]:
            extra_info[i]["TimeLimit.truncated"] = 1.0
            for j in index_array:
                extra_info[i]["PLAYER_" + str(j) + "_draw"] = 1.0

        rewards_role['self'] = rewards_all[:, :1]
        rewards_role['oppo'] = rewards_all[:, 1:]

        dones_role['self'] = dones_all[:, :1]
        dones_role['oppo'] = dones_all[:, 1:]

        return rewards_role, dones_role, extra_info
    
    def reset_single(self, k):

        self.step_counter[k] = 0
        height = self.rng.uniform(self.lower_height, self.upper_height, self.total_num_agents)
        degree = self.rng.uniform(0, 2 * np.pi, self.total_num_agents)

        radius_p = self.rng.uniform(0, self.control_r, 1)
        radius_e = self.rng.uniform(self.control_r, self.precaution_r, 2)
        radius = np.concatenate([radius_p, radius_e])

        obs_all_single = np.zeros(3*self.total_num_agents)

        self.ongoing[k] = False

        for i in range(self.total_num_agents):
            obs_all_single[3*i] = radius[i] * np.cos(degree[i])
            obs_all_single[3*i+1] = radius[i] * np.sin(degree[i])
            obs_all_single[3*i+2] = height[i]

        self.obs_all_raw[k] = copy.deepcopy(obs_all_single)
        self.obs_all_raw_prev[k] = copy.deepcopy(obs_all_single)

    def reset(self):
        self.sub_time = 0
        for i in range(self.num_threads):
            self.reset_single(i)
        self.gen_obs_from_raw()
        if self.oppo_policy is not None:
            self.oppo_rnn_states = np.zeros((self.num_threads, 1, self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, 1, 1), dtype=np.float32)
        return copy.deepcopy(self.obs)       

    def get_reward(self):
        rewards_role, dones_role, info = self.get_reward_all()
        self.rewards = rewards_role['self']
        self.dones = dones_role['self']
        self.oppo_done = dones_role['oppo']
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

            if i < 1:
                vel_array[:, 3*i:3*i+3] = self.max_vel_p * action_norm
            else:
                vel_array[:, 3*i:3*i+3] = self.max_vel_e * action_norm

        self.obs_all_raw_prev = copy.deepcopy(self.obs_all_raw)
        if mask is not None:
            vel_array *= mask
        self.obs_all_raw += vel_array * self.sim_dt

    def step(self, act):
        test_time_start = time.time()
        action = self._reset_action(act)
        # action_p = defender_policy(copy.deepcopy(self.other_obs), copy.deepcopy(self.distances[:, 0, :]))
        action_p = defender_policy_new(copy.deepcopy(self.other_obs))
        action = np.concatenate([action_p, action], axis= -1)
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
        oppo_actions_env = np.concatenate([self.oppo_action[:, idx, :] for idx in range(1)], axis=1)

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
    envs=NvSymmetry(n_th)
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
    # envs.oppo_policy = multi_policy(1, policies)
    obs = envs.reset()
    print("reset obs = ", obs)
    ob_op = copy.deepcopy(envs.oppo_obs)
    other_obs = copy.deepcopy(envs.other_obs)

    team_track = []
    oppo_track = []
    other_track = []


    team_track.append([obs[0][i][0:3] for i in range(1)])
    oppo_track.append([ob_op[0][i][0:3] for i in range(1)])
    other_track.append([other_obs[0][i][0:3] for i in range(1)])

    # print("sub_role_size = ", envs.sub_role_shape)

    game_long = 2000

    # for i in range(10):
    #     obs = envs.reset()
    #     print(obs)

    start_time = time.time()
    for i in range(game_long):
        # action_interface = [np.array([0.5,0.5,0,0]), np.array([0.9,-0.2,0,0]), np.array([-0.5,0.5,0,0])]
        #action_interface = [np.array([0,0,0.5,0])]
        action_interface = np.concatenate([np.expand_dims(np.concatenate([envs.action_space[i].sample() for i in range(1)]), axis=0) for _ in range(envs.num_threads)], axis=0)
        # action_interface = np.expand_dims(np.concatenate(action_interface), axis=0)
        # print("act = ", action_interface[0])
        nxt, rew, don, inf = envs.step(action_interface)
        # if don[0].all():
        #     break
        team_track.append([nxt[0][i][0:3] for i in range(1)])
        # print("team_track = ", team_track)
        ob_op = copy.deepcopy(envs.oppo_obs)
        other_obs = copy.deepcopy(envs.other_obs)
        oppo_track.append([ob_op[0][i][0:3] for i in range(1)])
        other_track.append([other_obs[0][i][0:3] for i in range(1)])
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
    # plot_track(other_track, team_track)
    # plot_track(other_track, oppo_track)