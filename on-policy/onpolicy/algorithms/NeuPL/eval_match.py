import numpy as np
import torch
import copy

def _t2n(x):
    return x.detach().cpu().numpy()

class eval_match_uav:
    def __init__(self, p1_policies, p2_policies, envs_p1, p1_win_list = None, p2_win_list = None, early_crash = None, draw_list = None):
        self.p1_policies = p1_policies
        self.p2_policies = p2_policies
        self.p1_num = len(p1_policies)
        self.p2_num = len(p2_policies)
        self.win_prob_mat = np.zeros((self.p1_num, self.p2_num))
        self.lose_num_mat = np.zeros((self.p1_num, self.p2_num))
        self.win_num_mat = np.zeros((self.p1_num, self.p2_num))
        self.draw_mat = np.zeros((self.p1_num, self.p2_num))
        self.total_round_mat = np.zeros((self.p1_num, self.p2_num))
        self.total_reward_mat = np.zeros((self.p1_num, self.p2_num))
        self.envs = envs_p1
        self.episode_length = self.envs.episode_length
        if p1_win_list is None:
            self.p1_win_list = ["seize_evader_done", "out_of_range_done"]
        else: 
            self.p1_win_list = p1_win_list
        if p2_win_list is None:
            self.p2_win_list = ["enter_lockdown_zone_done"]
        else: 
            self.p2_win_list = p2_win_list
        if early_crash is None:
            self.early_crash = ["approaching_terrian_done", "approaching_ceiling_done", "safe_distance_done"]
        else: 
            self.early_crash = early_crash
        if draw_list is None:
            self.draw_list = ["TimeLimit.truncated"]
        else:
            self.draw_list = draw_list
        self.num_agents = self.envs.world.num_team + self.envs.world.num_oppo
        self.pursuer_num = self.envs.world.num_team
        self.evader_num = self.envs.world.num_oppo
        self.actor = self.p1_policies[0].actor

    def update_policy(self, p1_policies, p2_policies):
        self.p1_policies = p1_policies
        self.p2_policies = p2_policies
        self.p1_num = len(p1_policies)
        self.p2_num = len(p2_policies)
        self.win_prob_mat = np.zeros((self.p1_num, self.p2_num))
        self.lose_num_mat = np.zeros((self.p1_num, self.p2_num))
        self.win_num_mat = np.zeros((self.p1_num, self.p2_num))
        self.draw_mat = np.zeros((self.p1_num, self.p2_num))
        self.total_round_mat = np.zeros((self.p1_num, self.p2_num))
        self.total_reward_mat = np.zeros((self.p1_num, self.p2_num))


    def checkVictory(self, info):
        pursuer_win = 0
        evader_win = 0
        draw_f = 0
        round_counter = 1

        play_evader_str = "PLAYER_" + str(self.num_agents - 1) + "_"
        evader_early_dead_list = [play_evader_str + item for item in self.early_crash]
        pursuer_win_list = self.p1_win_list + evader_early_dead_list
        evader_win_list = self.p2_win_list
        draw_final_list = self.draw_list
        for k in info:
            if any(sub in k for sub in pursuer_win_list):
                pursuer_win += 1
                break
            if any(sub in k for sub in evader_win_list):
                evader_win += 1
                break
            if any(sub in k for sub in draw_final_list):
                draw_f += 1
                break
        reason_flag = round_counter - (evader_win + pursuer_win + draw_f)
        if reason_flag == 1:
            terminal_reason = "evader win r: pursuers all crash"
        evader_win += reason_flag
        if pursuer_win == 1:
            for k in info:
                if reason_flag == 1:
                    break
                for sub in pursuer_win_list:
                    if sub in k:
                        terminal_reason = "pursuer win r: " + sub
                        reason_flag = 1
        elif evader_win == 1:
            for k in info:
                if reason_flag == 1:
                    break
                for sub in evader_win_list:
                    if sub in k:
                        terminal_reason = "evader win r: " + sub
                        reason_flag = 1
        else:
            for k in info:
                if reason_flag == 1:
                    break
                for sub in draw_final_list:
                    if sub in k:
                        terminal_reason = "evader win r: " + sub
                        reason_flag = 1
        return pursuer_win, evader_win, draw_f, terminal_reason
    
    def track_recorder(self, inx_p1, inx_p2):
        # record a single track
        selected_p1_policy = self.p1_policies[inx_p1]
        self.envs.world.oppo_policy = self.p2_policies[inx_p2]
        eval_obs = self.envs.reset()
        eval_rnn_states = np.zeros((1, self.pursuer_num, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((1, self.pursuer_num, 1), dtype=np.float32)
        track_r = []
        track_r.append(eval_obs)

        for eval_step in range(self.episode_length):
            selected_p1_policy.actor.eval()
            eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), 1))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), 1))
            eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(self.pursuer_num)], axis=1)

            # Obser reward and next obs
            # print("action network:", eval_actions_env)
            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
            
            # print(eval_infos)
            if eval_dones[0].all():
                p_win, e_win, d_num, terminal_reason = self.checkVictory(eval_infos[0])
                print("track: pursuer {} vs evader {}, {}!".format(inx_p1, inx_p2, terminal_reason))
                break

            track_r.append(eval_obs)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
            eval_masks = np.ones((1, self.pursuer_num, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
        return track_r, terminal_reason

    
    def calu_win_prob(self, n_rollout_threads, total_episodes, inx_p1, inx_p2, gamma = None):
        if gamma is None:
            gamma = 1.0
        selected_p1_policy = self.p1_policies[inx_p1]
        self.envs.world.oppo_policy = self.p2_policies[inx_p2]
        eval_obs = self.envs.reset()
        total_pursuer_win = 0
        total_evader_win = 0
        total_draw = 0
        total_reward = 0
        total_round = 0
        thread_discount = np.ones(n_rollout_threads, dtype= float)
        reward_eps = np.zeros(n_rollout_threads)
        eval_rnn_states = np.zeros((n_rollout_threads, self.pursuer_num, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, self.pursuer_num, 1), dtype=np.float32)

        for episodes in range(total_episodes):
            for eval_step in range(self.episode_length):
                thread_discount /= gamma
                selected_p1_policy.actor.eval()
                eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(self.pursuer_num)], axis=1)

                # Obser reward and next obs
                # print("action network:", eval_actions_env)
                eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
                # print(eval_infos)
                for i in range(n_rollout_threads):
                    reward_eps[i] += thread_discount[i] * np.mean(eval_rewards[i][:][0])
                    # total_reward += np.mean(eval_rewards[i][:][0])
                    if eval_dones[i].all():
                        p_win, e_win, d_num, _ = self.checkVictory(eval_infos[i])
                        total_pursuer_win += p_win
                        total_evader_win += e_win
                        total_draw += d_num
                        total_round += 1
                        reward_eps[i] /= thread_discount[i]
                        total_reward += reward_eps[i]
                        reward_eps[i] = 0
                        thread_discount[i] = 1
                        # if self.all_args.use_mix_policy: 
                        #     self.eval_envs.world.oppo_policy.updata_index_channel(i)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                eval_masks = np.ones((n_rollout_threads, self.pursuer_num, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            print("pursuer {} vs evader {} :episodes: {}/{}".format(inx_p1, inx_p2, episodes, total_episodes))
        
        win_prob_p1 = total_pursuer_win / total_round
        return win_prob_p1, total_pursuer_win, total_draw, total_round, total_reward
    
    def get_win_prob_mat(self, n_rollout_threads, episode_num, sub_mat_inx = None, gamma = None):
        if sub_mat_inx is not None:
            p1_line = range(sub_mat_inx[0], sub_mat_inx[1])
            p2_line = range(sub_mat_inx[2], sub_mat_inx[3])
        else:
            p1_line = range(self.p1_num)
            p2_line = range(self.p2_num)
        for i in p1_line:
            for j in p2_line:
                win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp, total_reward_ = self.calu_win_prob(n_rollout_threads, episode_num, i ,j, gamma)
                self.win_prob_mat[i,j] = win_prob_temp
                self.win_num_mat[i,j] = total_pwin_temp
                self.draw_mat[i,j] = total_draw_temp
                self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                self.total_round_mat[i,j] = total_round_temp
                self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_reward_mat)
        mask_zero = self.total_round_mat > 0.5
        payoff_mat[mask_zero] = (self.total_reward_mat[mask_zero])/(self.total_round_mat[mask_zero])
        return payoff_mat

    def get_win_prob_with_mask(self, n_rollout_threads, episode_num, mask = None, gamma = None):
        if mask is None:
            mask = np.ones((self.p1_num, self.p2_num), dtype=bool)
        for i in range(self.p1_num):
            for j in range(self.p2_num):
                if mask[i][j]:
                    win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp, total_reward_ = self.calu_win_prob(n_rollout_threads, episode_num, i ,j, gamma)
                    self.win_prob_mat[i,j] = win_prob_temp
                    self.win_num_mat[i,j] = total_pwin_temp
                    self.draw_mat[i,j] = total_draw_temp
                    self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                    self.total_round_mat[i,j] = total_round_temp
                    self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_reward_mat)
        mask_zero = self.total_round_mat > 0.5
        payoff_mat[mask_zero] = (self.total_reward_mat[mask_zero])/(self.total_round_mat[mask_zero])
        return payoff_mat

    def get_track_array(self):
        track_array = []
        info_array = []
        for i in range(self.p1_num):
            sub_track_array = []
            sub_info_array = []
            for j in range(self.p2_num):
                sub_track , ex_info = self.track_recorder(i, j)
                sub_track_array.append(sub_track)
                sub_info_array.append(ex_info)
            track_array.append(sub_track_array)
            info_array.append(sub_info_array)
        
        return track_array, info_array


class eval_match_uav_symmetry:
    def __init__(self, policies, envs, win_list = None, lose_list = None, draw_list = None):
        self.policies = policies
        self.policy_num = len(policies)
        self.win_prob_mat = np.zeros((self.policy_num, self.policy_num))
        self.lose_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.win_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.draw_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))
        self.envs = envs
        self.episode_length = self.envs.episode_length
        if win_list is None:
            self.win_list = ["PLAYER_0_win_reward", "PLAYER_1_lose_reward"]
        else: 
            self.win_list = win_list
        if lose_list is None:
            self.lose_list = ["PLAYER_0_lose_reward", "PLAYER_1_win_reward"]
        else: 
            self.lose_list = lose_list
        if draw_list is None:
            self.draw_list = ["PLAYER_0_draw", "PLAYER_1_draw"]
        else:
            self.draw_list = draw_list
        self.actor = self.policies[0].actor

    def update_policy(self, policies):
        self.policies = policies
        self.policy_num = len(policies)
        self.win_prob_mat = np.zeros((self.policy_num, self.policy_num))
        self.lose_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.win_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.draw_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))

    def checkVictory(self, info):
        _win = 0
        _lose = 0
        draw_f = 0
        round_counter = 1

        win_list = self.win_list
        lose_list = self.lose_list
        draw_final_list = self.draw_list
        for k in info:
            if any(sub in k for sub in win_list):
                _win += 1
                break
            if any(sub in k for sub in lose_list):
                _lose += 1
                break
            if any(sub in k for sub in draw_final_list):
                draw_f += 1
                break
        draw_f += round_counter - (_win + _lose + draw_f)
        return _win, _lose, draw_f
    
    def track_recorder(self, n_rollout_threads, inx_p1, inx_p2):
        # record a single track
        selected_p1_policy = self.policies[inx_p1]
        self.envs.world.oppo_policy = self.policies[inx_p2]
        eval_obs = self.envs.reset()
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
        track_r = []
        track_r.append(eval_obs)

        for eval_step in range(self.episode_length):
            selected_p1_policy.actor.eval()
            eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
            eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

            # Obser reward and next obs
            # print("action network:", eval_actions_env)
            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
            
            

            track_r.append(eval_obs)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
            eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
        return track_r

    
    def calu_win_prob(self, n_rollout_threads, total_episodes, inx_p1, inx_p2):
        selected_p1_policy = self.policies[inx_p1]
        self.envs.world.oppo_policy = self.policies[inx_p2]
        eval_obs = self.envs.reset()
        total_pursuer_win = 0
        total_evader_win = 0
        total_draw = 0
        total_round = 0
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)

        for episodes in range(total_episodes):
            for eval_step in range(self.episode_length):
                selected_p1_policy.actor.eval()
                eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                # Obser reward and next obs
                # print("action network:", eval_actions_env)
                eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
                # print(eval_infos)
                for i in range(n_rollout_threads):
                    if eval_dones[i].all():
                        p_win, e_win, d_num = self.checkVictory(eval_infos[i])
                        total_pursuer_win += p_win
                        total_evader_win += e_win
                        total_draw += d_num
                        total_round += 1
                        # if self.all_args.use_mix_policy: 
                        #     self.eval_envs.world.oppo_policy.updata_index_channel(i)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            print("policy {} vs policy {} :episodes: {}/{}".format(inx_p1, inx_p2, episodes, total_episodes))
        
        win_prob_p1 = total_pursuer_win / total_round
        return win_prob_p1, total_pursuer_win, total_draw, total_round
    
    def get_win_prob_mat(self, n_rollout_threads, episode_num):
        for i in range(self.policy_num):
            for j in range(i):
                win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp = self.calu_win_prob(n_rollout_threads, episode_num, i ,j)
                self.win_prob_mat[i,j] = win_prob_temp
                self.win_num_mat[i,j] = total_pwin_temp
                self.draw_mat[i,j] = total_draw_temp
                self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                self.total_round_mat[i,j] = total_round_temp
        payoff_mat = (self.win_num_mat - self.lose_num_mat)/(n_rollout_threads* episode_num)
        return payoff_mat

    def get_win_prob_with_mask(self, n_rollout_threads, episode_num, mask = None):
        if mask is None:
            mask = np.ones((self.policy_num, self.policy_num), dtype=bool)
        for i in range(self.policy_num):
            for j in range(i):
                if mask[i][j]:
                    win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp = self.calu_win_prob(n_rollout_threads, episode_num, i ,j)
                    self.win_prob_mat[i,j] = win_prob_temp
                    self.win_num_mat[i,j] = total_pwin_temp
                    self.draw_mat[i,j] = total_draw_temp
                    self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                    self.total_round_mat[i,j] = total_round_temp
        payoff_mat = (self.win_num_mat - self.lose_num_mat)/(n_rollout_threads * episode_num)
        return payoff_mat

    def get_track_array(self, n_rollout_threads):
        track_array = []
        for i in range(self.policy_num):
            sub_track_array = []
            for j in range(self.policy_num):
                sub_track = self.track_recorder(n_rollout_threads, i, j)
                sub_track_array.append(sub_track)
            track_array.append(sub_track_array)
        
        return track_array
    
    def get_full_win_prob_mat(self, n_rollout_threads, episode_num):
        for i in range(self.policy_num):
            for j in range(self.policy_num):
                win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp = self.calu_win_prob(n_rollout_threads, episode_num, i ,j)
                self.win_prob_mat[i,j] = win_prob_temp
                self.win_num_mat[i,j] = total_pwin_temp
                self.draw_mat[i,j] = total_draw_temp
                self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                self.total_round_mat[i,j] = total_round_temp
        payoff_mat = (self.win_num_mat - self.lose_num_mat)/(n_rollout_threads* episode_num)
        return payoff_mat
    

class eval_match_ball:
    def __init__(self, policies, envs, win_list = None, lose_list = None, draw_list = None):
        self.policies = policies
        self.policy_num = len(policies)
        self.win_prob_mat = np.zeros((self.policy_num, self.policy_num))
        self.lose_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.win_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.draw_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_reward_mat = np.zeros((self.policy_num, self.policy_num))
        self.envs = envs
        self.episode_length = self.envs.episode_length
        if win_list is None:
            self.win_list = ["PLAYER_0_win_reward", "PLAYER_1_lose_reward"]
        else: 
            self.win_list = win_list
        if lose_list is None:
            self.lose_list = ["PLAYER_0_lose_reward", "PLAYER_1_win_reward"]
        else: 
            self.lose_list = lose_list
        if draw_list is None:
            self.draw_list = ["PLAYER_0_draw", "PLAYER_1_draw"]
        else:
            self.draw_list = draw_list
        self.actor = self.policies[0].actor

    def update_policy(self, policies):
        self.policies = policies
        self.policy_num = len(policies)
        self.win_prob_mat = np.zeros((self.policy_num, self.policy_num))
        self.lose_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.win_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.draw_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_reward_mat = np.zeros((self.policy_num, self.policy_num))

    def checkVictory(self, info):
        _win = 0
        _lose = 0
        draw_f = 0
        round_counter = 1

        win_list = self.win_list
        lose_list = self.lose_list
        draw_final_list = self.draw_list
        for k in info:
            if any(sub in k for sub in win_list):
                _win += 1
                break
            if any(sub in k for sub in lose_list):
                _lose += 1
                break
            if any(sub in k for sub in draw_final_list):
                draw_f += 1
                break
        draw_f += round_counter - (_win + _lose + draw_f)
        return _win, _lose, draw_f
    
    def track_recorder(self, n_rollout_threads, inx_p1, inx_p2):
        # record a single track
        selected_p1_policy = self.policies[inx_p1]
        self.envs.world.oppo_policy = self.policies[inx_p2]
        eval_obs = self.envs.reset()
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
        track_r = []
        track_r.append(eval_obs)

        for eval_step in range(self.episode_length):
            selected_p1_policy.actor.eval()
            eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
            eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

            # Obser reward and next obs
            # print("action network:", eval_actions_env)
            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
            
            

            track_r.append(eval_obs)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
            eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
        return track_r

    
    def calu_win_prob(self, n_rollout_threads, total_episodes, inx_p1, inx_p2):
        selected_p1_policy = self.policies[inx_p1]
        self.envs.world.oppo_policy = self.policies[inx_p2]
        eval_obs = self.envs.reset()
        total_pursuer_win = 0
        total_evader_win = 0
        total_draw = 0
        total_round = 0
        total_reward = 0
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)

        for episodes in range(total_episodes):
            for eval_step in range(self.episode_length):
                selected_p1_policy.actor.eval()
                eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                # Obser reward and next obs
                # print("action network:", eval_actions_env)
                eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
                # print(eval_infos)
                for i in range(n_rollout_threads):
                    total_reward += eval_rewards[i][0][0]
                    if eval_dones[i].all():
                        p_win, e_win, d_num = self.checkVictory(eval_infos[i])
                        total_pursuer_win += p_win
                        total_evader_win += e_win
                        total_draw += d_num
                        total_round += 1
                        # if self.all_args.use_mix_policy: 
                        #     self.eval_envs.world.oppo_policy.updata_index_channel(i)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            print("policy {} vs policy {} :episodes: {}/{}".format(inx_p1, inx_p2, episodes, total_episodes))
        
        win_prob_p1 = total_pursuer_win / total_round
        return win_prob_p1, total_pursuer_win, total_draw, total_round, total_reward
    
    def get_win_prob_mat(self, n_rollout_threads, episode_num):
        for i in range(self.policy_num):
            for j in range(i):
                win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp, total_reward_ = self.calu_win_prob(n_rollout_threads, episode_num, i ,j)
                self.win_prob_mat[i,j] = win_prob_temp
                self.win_num_mat[i,j] = total_pwin_temp
                self.draw_mat[i,j] = total_draw_temp
                self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                self.total_round_mat[i,j] = total_round_temp
                self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_round_mat)
        mask_positive = self.total_round_mat > 0.5
        payoff_mat[mask_positive] = (self.total_reward_mat[mask_positive])/(self.total_round_mat[mask_positive])
        return payoff_mat

    def get_win_prob_with_mask(self, n_rollout_threads, episode_num, mask = None):
        if mask is None:
            mask = np.ones((self.policy_num, self.policy_num), dtype=bool)
        for i in range(self.policy_num):
            for j in range(i):
                if mask[i][j]:
                    win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp, total_reward_ = self.calu_win_prob(n_rollout_threads, episode_num, i ,j)
                    self.win_prob_mat[i,j] = win_prob_temp
                    self.win_num_mat[i,j] = total_pwin_temp
                    self.draw_mat[i,j] = total_draw_temp
                    self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                    self.total_round_mat[i,j] = total_round_temp
                    self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_round_mat)
        mask_positive = self.total_round_mat > 0.5
        payoff_mat[mask_positive] = (self.total_reward_mat[mask_positive])/(self.total_round_mat[mask_positive])
        return payoff_mat

    def get_track_array(self, n_rollout_threads):
        track_array = []
        for i in range(self.policy_num):
            sub_track_array = []
            for j in range(self.policy_num):
                sub_track = self.track_recorder(n_rollout_threads, i, j)
                sub_track_array.append(sub_track)
            track_array.append(sub_track_array)
        
        return track_array

class eval_ball_cross_match:
    def __init__(self, policies, opponent_policies, envs, win_list = None, lose_list = None, draw_list = None):
        self.policies = policies
        self.opponent_policies = opponent_policies
        self.p1_num = len(policies)
        self.p2_num = len(opponent_policies)
        self.win_prob_mat = np.zeros((self.p1_num, self.p2_num))
        self.lose_num_mat = np.zeros((self.p1_num, self.p2_num))
        self.win_num_mat = np.zeros((self.p1_num, self.p2_num))
        self.draw_mat = np.zeros((self.p1_num, self.p2_num))
        self.total_round_mat = np.zeros((self.p1_num, self.p2_num))
        self.envs = envs
        self.episode_length = self.envs.episode_length
        if win_list is None:
            self.win_list = ["PLAYER_0_win_reward", "PLAYER_1_lose_reward"]
        else: 
            self.win_list = win_list
        if lose_list is None:
            self.lose_list = ["PLAYER_0_lose_reward", "PLAYER_1_win_reward"]
        else: 
            self.lose_list = lose_list
        if draw_list is None:
            self.draw_list = ["PLAYER_0_draw", "PLAYER_1_draw"]
        else:
            self.draw_list = draw_list
        self.actor = self.policies[0].actor

    def update_policy(self, policies, opponent_policies):
        self.policies = policies
        self.opponent_policies = opponent_policies
        self.p1_num = len(policies)
        self.p2_num = len(opponent_policies)
        self.win_prob_mat = np.zeros((self.p1_num, self.p2_num))
        self.lose_num_mat = np.zeros((self.p1_num, self.p2_num))
        self.win_num_mat = np.zeros((self.p1_num, self.p2_num))
        self.draw_mat = np.zeros((self.p1_num, self.p2_num))
        self.total_round_mat = np.zeros((self.p1_num, self.p2_num))

    def checkVictory(self, info):
        _win = 0
        _lose = 0
        draw_f = 0
        round_counter = 1

        win_list = self.win_list
        lose_list = self.lose_list
        draw_final_list = self.draw_list
        for k in info:
            if any(sub in k for sub in win_list):
                _win += 1
                break
            if any(sub in k for sub in lose_list):
                _lose += 1
                break
            if any(sub in k for sub in draw_final_list):
                draw_f += 1
                break
        draw_f += round_counter - (_win + _lose + draw_f)
        return _win, _lose, draw_f
    
    def track_recorder(self, n_rollout_threads, inx_p1, inx_p2):
        # record a single track
        selected_p1_policy = self.policies[inx_p1]
        self.envs.world.oppo_policy = self.opponent_policies[inx_p2]
        eval_obs = self.envs.reset()
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
        track_r = []
        done_r = []
        track_r.append(eval_obs)

        for eval_step in range(self.episode_length):
            selected_p1_policy.actor.eval()
            eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
            eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

            # Obser reward and next obs
            # print("action network:", eval_actions_env)
            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
            
            

            track_r.append(eval_obs)
            done_r.append(eval_dones)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
            eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
        return np.array(track_r), np.array(done_r)

    
    def calu_win_prob(self, n_rollout_threads, total_episodes, inx_p1, inx_p2):
        selected_p1_policy = self.policies[inx_p1]
        self.envs.world.oppo_policy = self.opponent_policies[inx_p2]
        eval_obs = self.envs.reset()
        total_pursuer_win = 0
        total_evader_win = 0
        total_draw = 0
        total_round = 0
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)

        for episodes in range(total_episodes):
            for eval_step in range(self.episode_length):
                selected_p1_policy.actor.eval()
                eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                # Obser reward and next obs
                # print("action network:", eval_actions_env)
                eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
                # print(eval_infos)
                for i in range(n_rollout_threads):
                    if eval_dones[i].all():
                        p_win, e_win, d_num = self.checkVictory(eval_infos[i])
                        total_pursuer_win += p_win
                        total_evader_win += e_win
                        total_draw += d_num
                        total_round += 1
                        # if self.all_args.use_mix_policy: 
                        #     self.eval_envs.world.oppo_policy.updata_index_channel(i)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            print("policy {} vs policy {} :episodes: {}/{}".format(inx_p1, inx_p2, episodes, total_episodes))
        
        win_prob_p1 = total_pursuer_win / total_round
        return win_prob_p1, total_pursuer_win, total_draw, total_round
    
    def get_win_prob_mat(self, n_rollout_threads, episode_num):
        for i in range(self.p1_num):
            for j in range(self.p2_num):
                win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp = self.calu_win_prob(n_rollout_threads, episode_num, i ,j)
                self.win_prob_mat[i,j] = win_prob_temp
                self.win_num_mat[i,j] = total_pwin_temp
                self.draw_mat[i,j] = total_draw_temp
                self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                self.total_round_mat[i,j] = total_round_temp
        payoff_mat = (self.win_num_mat - self.lose_num_mat)/(n_rollout_threads* episode_num)
        return payoff_mat

    def get_win_prob_with_mask(self, n_rollout_threads, episode_num, mask = None):
        if mask is None:
            mask = np.ones((self.p1_num, self.p2_num), dtype=bool)
        for i in range(self.p1_num):
            for j in range(self.p2_num):
                if mask[i][j]:
                    win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp = self.calu_win_prob(n_rollout_threads, episode_num, i ,j)
                    self.win_prob_mat[i,j] = win_prob_temp
                    self.win_num_mat[i,j] = total_pwin_temp
                    self.draw_mat[i,j] = total_draw_temp
                    self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                    self.total_round_mat[i,j] = total_round_temp
        payoff_mat = (self.win_num_mat - self.lose_num_mat)/(n_rollout_threads * episode_num)
        return payoff_mat

    def get_track_array(self, n_rollout_threads, mask = None):
        if mask is None:
            mask  = np.ones((self.p1_num, self.p2_num), dtype= bool)
        track_array = []
        done_array = []
        for i in range(self.p1_num):
            sub_track_array = []
            sub_done_array = []
            for j in range(self.p2_num):
                if mask[i][j]:
                    sub_track, sub_done = self.track_recorder(n_rollout_threads, i, j)
                else:
                    sub_track, sub_done = None, None
                sub_track_array.append(sub_track)
                sub_done_array.append(sub_done)
            track_array.append(sub_track_array)
            done_array.append(sub_done_array)
        
        return track_array, done_array
    
class eval_match_ball_final:
    def __init__(self, policies, envs, policy_type = None, win_list = None, lose_list = None, draw_list = None):
        self.policies = policies
        self.policy_num = len(policies)
        self.win_prob_mat = np.zeros((self.policy_num, self.policy_num))
        self.lose_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.win_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.draw_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_reward_mat = np.zeros((self.policy_num, self.policy_num))
        self.envs = envs
        if policy_type is None:
            self.policy_type = dict()
            for i in range(self.policy_num):
                self.policy_type[i] = "pure"
        else:
            self.policy_type = policy_type
        self.episode_length = self.envs.episode_length
        if win_list is None:
            self.win_list = ["PLAYER_0_win_reward", "PLAYER_1_lose_reward"]
        else: 
            self.win_list = win_list
        if lose_list is None:
            self.lose_list = ["PLAYER_0_lose_reward", "PLAYER_1_win_reward"]
        else: 
            self.lose_list = lose_list
        if draw_list is None:
            self.draw_list = ["PLAYER_0_draw", "PLAYER_1_draw"]
        else:
            self.draw_list = draw_list
        self.actor = self.policies[0].actor

    def update_policy(self, policies, policy_type = None):
        self.policies = policies
        self.policy_num = len(policies)
        self.win_prob_mat = np.zeros((self.policy_num, self.policy_num))
        self.lose_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.win_num_mat = np.zeros((self.policy_num, self.policy_num))
        self.draw_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_reward_mat = np.zeros((self.policy_num, self.policy_num))
        if policy_type is None:
            self.policy_type = dict()
            for i in range(self.policy_num):
                self.policy_type[i] = "pure"
        else:
            self.policy_type = policy_type
        self.actor = self.policies[0].actor

    def simple_policy_eval(self, eval_eps, n_rollout_threads, policies, oppo_policies = None, policy_type = None, oppo_policy_type = None, delta = None):
        n_p1 = len(policies)
        if policy_type is None:
            policy_type = dict()
            for i in range(n_p1):
                policy_type[i] = "pure"

        if oppo_policies is None:
            oppo_policies = copy.deepcopy(policies)
            oppo_policy_type = copy.deepcopy(policy_type)

        n_p2 = len(oppo_policies)

        if oppo_policy_type is None:
            oppo_policy_type = dict()
            for i in range(n_p2):
                oppo_policy_type[i] = "pure"

        if delta is None:
            delta = np.inf

        rewards_mat = np.zeros((n_p1, n_p2))
        round_mat = np.zeros((n_p1, n_p2))
        std_mat = np.zeros((n_p1, n_p2))

        for p1_i in range(n_p1):
            for p2_i in range(n_p2):
                selected_p1_policy = copy.deepcopy(policies[p1_i])
                self.envs.world.oppo_policy = copy.deepcopy(oppo_policies[p2_i])
                eval_obs = self.envs.reset()
                total_round = 0
                total_reward = 0
                eva_r_list = []
                eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)

                while True:
                    for episodes in range(eval_eps):
                        for eval_step in range(self.episode_length):
                            selected_p1_policy.actor.eval()
                            eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                                np.concatenate(eval_rnn_states),
                                                                np.concatenate(eval_masks),
                                                                deterministic=True)
                            eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                            eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                            # Obser reward and next obs
                            # print("action network:", eval_actions_env)
                            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
                            # print(eval_infos)
                            for i in range(n_rollout_threads):
                                total_reward += eval_rewards[i][0][0]
                                if eval_dones[i].all():
                                    total_round += 1
                                    eva_r_list.append(eval_rewards[i][0][0])
                                    if policy_type[p1_i] == "mixed": 
                                        selected_p1_policy.update_index_channel(i)
                                    if oppo_policy_type[p2_i] == "mixed":
                                        self.envs.world.oppo_policy.update_index_channel(i)

                            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                            eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                        
                        # print("policy {} vs policy {} :episodes: {}/{}".format(p1_i, p2_i, episodes, eval_eps))
                    
                    std_ = np.std(np.array(eva_r_list))/np.sqrt(len(eva_r_list))
                    print("standard vaule of match: {} vs {} = {}, delta = {}".format(p1_i, p2_i, std_, delta))
                    if std_ < delta:
                        break

                round_mat[p1_i, p2_i] = total_round
                rewards_mat[p1_i, p2_i] = total_reward
                std_mat[p1_i, p2_i] = np.std(np.array(eva_r_list))/np.sqrt(len(eva_r_list))

        payoff_mat = rewards_mat / round_mat
                
        return payoff_mat, std_mat

    def checkVictory(self, info):
        _win = 0
        _lose = 0
        draw_f = 0
        round_counter = 1

        win_list = self.win_list
        lose_list = self.lose_list
        draw_final_list = self.draw_list
        for k in info:
            if any(sub in k for sub in win_list):
                _win += 1
                break
            if any(sub in k for sub in lose_list):
                _lose += 1
                break
            if any(sub in k for sub in draw_final_list):
                draw_f += 1
                break
        draw_f += round_counter - (_win + _lose + draw_f)
        return _win, _lose, draw_f
    
    def track_recorder(self, n_rollout_threads, inx_p1, inx_p2):
        # record a single track
        selected_p1_policy = copy.deepcopy(self.policies[inx_p1])
        self.envs.world.oppo_policy = copy.deepcopy(self.policies[inx_p2])
        eval_obs = self.envs.reset()
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
        track_r = []
        track_r.append(eval_obs)

        for eval_step in range(self.episode_length):
            selected_p1_policy.actor.eval()
            eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
            eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

            # Obser reward and next obs
            # print("action network:", eval_actions_env)
            eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
            
            for i in range(n_rollout_threads):
                    if eval_dones[i].all():
                        if self.policy_type[inx_p1] == "mixed": 
                            selected_p1_policy.updata_index_channel(i)
                        if self.policy_type[inx_p2] == "mixed":
                            self.envs.world.oppo_policy.updata_index_channel(i)

            track_r.append(eval_obs)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
            eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        
        return track_r

    
    def calu_win_prob(self, n_rollout_threads, total_episodes, inx_p1, inx_p2, delta = None):
        selected_p1_policy = copy.deepcopy(self.policies[inx_p1])
        self.envs.world.oppo_policy = copy.deepcopy(self.policies[inx_p2])
        eval_obs = self.envs.reset()
        total_pursuer_win = 0
        total_evader_win = 0
        total_draw = 0
        total_round = 0
        total_reward = 0
        eval_r_list = []
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
        if delta is None:
            delta = np.inf

        while True:
            for episodes in range(total_episodes):
                for eval_step in range(self.episode_length):
                    selected_p1_policy.actor.eval()
                    eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                        np.concatenate(eval_rnn_states),
                                                        np.concatenate(eval_masks),
                                                        deterministic=True)
                    eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                    eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                    eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                    # Obser reward and next obs
                    # print("action network:", eval_actions_env)
                    eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
                    # print(eval_infos)
                    for i in range(n_rollout_threads):
                        total_reward += eval_rewards[i][0][0]
                        if eval_dones[i].all():
                            eval_r_list.append(eval_rewards[i][0][0])
                            p_win, e_win, d_num = self.checkVictory(eval_infos[i])
                            total_pursuer_win += p_win
                            total_evader_win += e_win
                            total_draw += d_num
                            total_round += 1
                            if self.policy_type[inx_p1] == "mixed": 
                                selected_p1_policy.update_index_channel(i)
                            if self.policy_type[inx_p2] == "mixed": 
                                self.envs.world.oppo_policy.update_index_channel(i)

                    eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                    eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                    eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                
                # print("policy {} vs policy {} :episodes: {}/{}".format(inx_p1, inx_p2, episodes, total_episodes))

            std_ = np.std(np.array(eval_r_list))/np.sqrt(len(eval_r_list))
            payoff_ = total_reward/total_round
            print("policy {} vs policy {} : payoff {}, std {}, delta {}".format(inx_p1, inx_p2, payoff_, std_, delta))
            if std_ < delta:
                break
        
        win_prob_p1 = total_pursuer_win / total_round
        return win_prob_p1, total_pursuer_win, total_draw, total_round, total_reward
    
    def calu_win_prob_N(self, n_rollout_threads, min_sample_N, inx_p1, inx_p2):
        selected_p1_policy = copy.deepcopy(self.policies[inx_p1])
        self.envs.world.oppo_policy = copy.deepcopy(self.policies[inx_p2])
        eval_obs = self.envs.reset()
        total_pursuer_win = 0
        total_evader_win = 0
        total_draw = 0
        total_round = 0
        total_reward = 0
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)

        while total_round < min_sample_N:
            for eval_step in range(self.episode_length):
                selected_p1_policy.actor.eval()
                eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                # Obser reward and next obs
                # print("action network:", eval_actions_env)
                eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
                # print(eval_infos)
                for i in range(n_rollout_threads):
                    total_reward += eval_rewards[i][0][0]
                    if eval_dones[i].all():
                        p_win, e_win, d_num = self.checkVictory(eval_infos[i])
                        total_pursuer_win += p_win
                        total_evader_win += e_win
                        total_draw += d_num
                        total_round += 1
                        if self.policy_type[inx_p1] == "mixed": 
                            selected_p1_policy.update_index_channel(i)
                        if self.policy_type[inx_p2] == "mixed": 
                            self.envs.world.oppo_policy.update_index_channel(i)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            print("policy {} vs policy {} :episodes: {}/{}".format(inx_p1, inx_p2, total_round, min_sample_N))
        
        win_prob_p1 = total_pursuer_win / total_round
        return win_prob_p1, total_pursuer_win, total_draw, total_round, total_reward
    
    def get_win_prob_mat(self, n_rollout_threads, episode_num):
        for i in range(self.policy_num):
            for j in range(i):
                win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp, total_reward_ = self.calu_win_prob(n_rollout_threads, episode_num, i ,j)
                self.win_prob_mat[i,j] = win_prob_temp
                self.win_num_mat[i,j] = total_pwin_temp
                self.draw_mat[i,j] = total_draw_temp
                self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                self.total_round_mat[i,j] = total_round_temp
                self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_round_mat)
        mask_positive = self.total_round_mat > 0.5
        payoff_mat[mask_positive] = (self.total_reward_mat[mask_positive])/(self.total_round_mat[mask_positive])
        return payoff_mat
    
    def get_win_prob_mat_N(self, n_rollout_threads, min_simple_N):
        for i in range(self.policy_num):
            for j in range(i):
                win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp, total_reward_ = self.calu_win_prob_N(n_rollout_threads, min_simple_N, i ,j)
                self.win_prob_mat[i,j] = win_prob_temp
                self.win_num_mat[i,j] = total_pwin_temp
                self.draw_mat[i,j] = total_draw_temp
                self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                self.total_round_mat[i,j] = total_round_temp
                self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_round_mat)
        mask_positive = self.total_round_mat > 0.5
        payoff_mat[mask_positive] = (self.total_reward_mat[mask_positive])/(self.total_round_mat[mask_positive])
        return payoff_mat

    def get_win_prob_with_mask(self, n_rollout_threads, episode_num, mask = None, delta_mat = None):
        if mask is None:
            mask = np.ones((self.policy_num, self.policy_num), dtype=bool)
        for i in range(self.policy_num):
            for j in range(i):
                if mask[i][j]:
                    win_prob_temp, total_pwin_temp, total_draw_temp,total_round_temp, total_reward_ = self.calu_win_prob(n_rollout_threads, episode_num, i ,j, delta=None if delta_mat is None else delta_mat[i,j])
                    self.win_prob_mat[i,j] = win_prob_temp
                    self.win_num_mat[i,j] = total_pwin_temp
                    self.draw_mat[i,j] = total_draw_temp
                    self.lose_num_mat[i,j] = total_round_temp - total_draw_temp - total_pwin_temp
                    self.total_round_mat[i,j] = total_round_temp
                    self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_round_mat)
        mask_positive = self.total_round_mat > 0.5
        payoff_mat[mask_positive] = (self.total_reward_mat[mask_positive])/(self.total_round_mat[mask_positive])
        return payoff_mat

    def get_track_array(self, n_rollout_threads):
        track_array = []
        for i in range(self.policy_num):
            sub_track_array = []
            for j in range(self.policy_num):
                sub_track = self.track_recorder(n_rollout_threads, i, j)
                sub_track_array.append(sub_track)
            track_array.append(sub_track_array)
        
        return track_array
    
class eval_match_poker:
    def __init__(self, policies, envs, policy_type = None):
        self.policies = policies
        self.policy_num = len(policies)
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_reward_mat = np.zeros((self.policy_num, self.policy_num))
        self.envs = envs
        if policy_type is None:
            self.policy_type = dict()
            for i in range(self.policy_num):
                self.policy_type[i] = "pure"
        else:
            self.policy_type = policy_type
        self.episode_length = self.envs.episode_length
        self.actor = self.policies[0].actor

    def update_policy(self, policies, policy_type = None):
        self.policies = policies
        self.policy_num = len(policies)
        self.total_round_mat = np.zeros((self.policy_num, self.policy_num))
        self.total_reward_mat = np.zeros((self.policy_num, self.policy_num))
        if policy_type is None:
            self.policy_type = dict()
            for i in range(self.policy_num):
                self.policy_type[i] = "pure"
        else:
            self.policy_type = policy_type
        self.actor = self.policies[0].actor
    
    def calu_payoff(self, n_rollout_threads, total_episodes, inx_p1, inx_p2, delta = None):
        selected_p1_policy = copy.deepcopy(self.policies[inx_p1])
        self.envs.world.oppo_policy = copy.deepcopy(self.policies[inx_p2])
        eval_obs, eval_a_acts = self.envs.reset()
        total_round = 0
        total_reward = 0
        eval_r_list = []
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
        if delta is None:
            delta = np.inf

        while True:
            for episodes in range(total_episodes):
                for eval_step in range(self.episode_length):
                    selected_p1_policy.actor.eval()
                    eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                        np.concatenate(eval_rnn_states),
                                                        np.concatenate(eval_masks),
                                                        np.concatenate(eval_a_acts),
                                                        deterministic=True)
                    eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                    eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                    eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                    # Obser reward and next obs
                    # print("action network:", eval_actions_env)
                    eval_obs, eval_rewards, eval_dones, eval_infos, eval_a_acts = self.envs.step(eval_actions_env)
                    # print(eval_infos)
                    for i in range(n_rollout_threads):
                        total_reward += eval_rewards[i][0][0]
                        if eval_dones[i].all():
                            eval_r_list.append(eval_rewards[i][0][0])
                            total_round += 1
                            if self.policy_type[inx_p1] == "mixed": 
                                selected_p1_policy.update_index_channel(i)
                            if self.policy_type[inx_p2] == "mixed": 
                                self.envs.world.oppo_policy.update_index_channel(i)

                    eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                    eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                    eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            std_ = np.std(np.array(eval_r_list))/np.sqrt(len(eval_r_list))
            payoff_ = total_reward/total_round
            print("policy {} vs policy {} : payoff {}, std {}, delta {}".format(inx_p1, inx_p2, payoff_, std_, delta))
            if std_ < delta:
                break
        
        return total_round, total_reward
    
    def calu_payoff_N(self, n_rollout_threads, min_sample_N, inx_p1, inx_p2):
        selected_p1_policy = copy.deepcopy(self.policies[inx_p1])
        self.envs.world.oppo_policy = copy.deepcopy(self.policies[inx_p2])
        eval_obs, eval_a_acts = self.envs.reset()
        total_round = 0
        total_reward = 0
        eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
        eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)

        while total_round < min_sample_N:
            for eval_step in range(self.episode_length):
                selected_p1_policy.actor.eval()
                eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    np.concatenate(eval_a_acts),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                # Obser reward and next obs
                # print("action network:", eval_actions_env)
                eval_obs, eval_rewards, eval_dones, eval_infos, eval_a_acts = self.envs.step(eval_actions_env)
                # print(eval_infos)
                for i in range(n_rollout_threads):
                    total_reward += eval_rewards[i][0][0]
                    if eval_dones[i].all():
                        total_round += 1
                        if self.policy_type[inx_p1] == "mixed": 
                            selected_p1_policy.update_index_channel(i)
                        if self.policy_type[inx_p2] == "mixed": 
                            self.envs.world.oppo_policy.update_index_channel(i)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            print("policy {} vs policy {} :episodes: {}/{}".format(inx_p1, inx_p2, total_round, min_sample_N))
        
        return total_round, total_reward
    
    def simple_policy_eval(self, eval_eps, n_rollout_threads, policies, oppo_policies = None, policy_type = None, oppo_policy_type = None, delta = None):
        n_p1 = len(policies)
        if policy_type is None:
            policy_type = dict()
            for i in range(n_p1):
                policy_type[i] = "pure"

        if oppo_policies is None:
            oppo_policies = copy.deepcopy(policies)
            oppo_policy_type = copy.deepcopy(policy_type)

        n_p2 = len(oppo_policies)

        if oppo_policy_type is None:
            oppo_policy_type = dict()
            for i in range(n_p2):
                oppo_policy_type[i] = "pure"

        rewards_mat = np.zeros((n_p1, n_p2))
        round_mat = np.zeros((n_p1, n_p2))
        std_mat = np.zeros((n_p1, n_p2))
        if delta is None:
            delta = np.inf

        for p1_i in range(n_p1):
            for p2_i in range(n_p2):
                selected_p1_policy = copy.deepcopy(policies[p1_i])
                self.envs.world.oppo_policy = copy.deepcopy(oppo_policies[p2_i])
                eval_obs, eval_a_acts = self.envs.reset()
                total_round = 0
                total_reward = 0
                eva_r_list = []
                eval_rnn_states = np.zeros((n_rollout_threads, 1, self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)

                while True:
                    for episodes in range(eval_eps):
                        for eval_step in range(self.episode_length):
                            selected_p1_policy.actor.eval()
                            eval_action, eval_rnn_states = selected_p1_policy.act(np.concatenate(eval_obs),
                                                                np.concatenate(eval_rnn_states),
                                                                np.concatenate(eval_masks),
                                                                np.concatenate(eval_a_acts),
                                                                deterministic=True)
                            eval_actions = np.array(np.split(_t2n(eval_action), n_rollout_threads))
                            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), n_rollout_threads))
                            eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(1)], axis=1)

                            # Obser reward and next obs
                            # print("action network:", eval_actions_env)
                            eval_obs, eval_rewards, eval_dones, eval_infos, eval_a_acts = self.envs.step(eval_actions_env)
                            # print(eval_infos)
                            for i in range(n_rollout_threads):
                                total_reward += eval_rewards[i][0][0]
                                if eval_dones[i].all():
                                    total_round += 1
                                    eva_r_list.append(eval_rewards[i][0][0])
                                    if policy_type[p1_i] == "mixed": 
                                        selected_p1_policy.update_index_channel(i)
                                    if oppo_policy_type[p2_i] == "mixed": 
                                        self.envs.world.oppo_policy.update_index_channel(i)

                            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.actor._recurrent_N, self.actor.hidden_size), dtype=np.float32)
                            eval_masks = np.ones((n_rollout_threads, 1, 1), dtype=np.float32)
                            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
                        
                        # print("policy {} vs policy {} :episodes: {}/{}".format(p1_i, p2_i, episodes, eval_eps))
                    std_ = np.std(np.array(eva_r_list))/np.sqrt(len(eva_r_list))
                    print("standard vaule of match: {} vs {} = {}, delta = {}".format(p1_i, p2_i, std_, delta))
                    if std_ < delta:
                        break

                round_mat[p1_i, p2_i] = total_round
                rewards_mat[p1_i, p2_i] = total_reward
                std_mat[p1_i, p2_i] = np.std(np.array(eva_r_list))/np.sqrt(len(eva_r_list))

        payoff_mat = rewards_mat / round_mat
                
        return payoff_mat, std_mat

    
    def get_win_prob_mat(self, n_rollout_threads, episode_num, delta = None):
        for i in range(self.policy_num):
            for j in range(i):
                total_round_temp, total_reward_ = self.calu_payoff(n_rollout_threads, episode_num, i ,j, delta=delta)
                self.total_round_mat[i,j] = total_round_temp
                self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_round_mat)
        mask_positive = self.total_round_mat > 0.5
        payoff_mat[mask_positive] = (self.total_reward_mat[mask_positive])/(self.total_round_mat[mask_positive])
        return payoff_mat
    
    def get_win_prob_mat_N(self, n_rollout_threads, min_simple_N):
        for i in range(self.policy_num):
            for j in range(i):
                total_round_temp, total_reward_ = self.calu_payoff_N(n_rollout_threads, min_simple_N, i ,j)
                self.total_round_mat[i,j] = total_round_temp
                self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_round_mat)
        mask_positive = self.total_round_mat > 0.5
        payoff_mat[mask_positive] = (self.total_reward_mat[mask_positive])/(self.total_round_mat[mask_positive])
        return payoff_mat

    def get_win_prob_with_mask(self, n_rollout_threads, episode_num, mask = None, delta_mat = None):
        if mask is None:
            mask = np.ones((self.policy_num, self.policy_num), dtype=bool)
        for i in range(self.policy_num):
            for j in range(i):
                if mask[i][j]:
                    total_round_temp, total_reward_ = self.calu_payoff(n_rollout_threads, episode_num, i ,j, delta=None if delta_mat is None else delta_mat[i,j])
                    self.total_round_mat[i,j] = total_round_temp
                    self.total_reward_mat[i,j] = total_reward_
        payoff_mat = np.zeros_like(self.total_round_mat)
        mask_positive = self.total_round_mat > 0.5
        payoff_mat[mask_positive] = (self.total_reward_mat[mask_positive])/(self.total_round_mat[mask_positive])
        return payoff_mat
