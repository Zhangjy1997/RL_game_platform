# TODO:

import numpy as np
import torch
import wandb
from onpolicy.algorithms.NeuPL.eval_match import eval_match_uav as eval_match
from onpolicy.algorithms.NeuPL.Policy_prob_matrix import Nash_matrix as prob_matrix
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as mixing_policy
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as Policy
import random
import os
import re
import time
import copy
import json

def restore_eval_policy(policy, model_dir, label_str, use_mixer = True, head_str = None):
    """Restore policy's networks from a saved model."""
    if head_str is None:
        policy_actor_state_dict = torch.load(str(model_dir) + '/actor_' + label_str + '.pt')
        policy.actor.load_state_dict(policy_actor_state_dict)
        if use_mixer:
            policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer_' + label_str + '.pt')
            policy.mixer.load_state_dict(policy_mixer_state_dict)
    else:
        policy_actor_state_dict = torch.load(str(model_dir)  + '/actor_' + str(head_str)+ '_' + label_str + '.pt')
        policy.actor.load_state_dict(policy_actor_state_dict)
        if use_mixer:
            policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer_' + str(head_str)+ '_' + label_str + '.pt')
            policy.mixer.load_state_dict(policy_mixer_state_dict)

def creat_policy(args, envs, save_dir, label_str, device, use_mixer):
    policy = Policy(args,
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_act_space[0],
                        device)
    restore_eval_policy(policy, save_dir, label_str, use_mixer)
    return policy

def check_convergence(payoffs, deltas, coeff_std, N):
    """
    Check if the coeff_std*delta neighborhoods of the top N payoffs have an intersection.

    Args:
    - payoffs (list or np.array): List of payoffs.
    - deltas (list or np.array): List of standard deviations corresponding to the payoffs.
    - N (int): Number of top elements to consider.

    Returns:
    - bool: True if there is an intersection, False otherwise.
    """

    # Check if lengths of payoffs and deltas are equal
    if len(payoffs) != len(deltas):
        raise ValueError("Length of payoffs and deltas must be the same.")

    # Sort payoffs and deltas based on payoffs
    sorted_indices = np.argsort(payoffs)[::-1]
    sorted_payoffs = np.array(payoffs)[sorted_indices]
    sorted_deltas = np.array(deltas)[sorted_indices]

    # Get the top N payoffs and their corresponding deltas
    top_payoffs = sorted_payoffs[:N]
    top_deltas = sorted_deltas[:N]

    # Calculate the intervals (neighborhoods)
    intervals = [(p - coeff_std * d, p + coeff_std * d) for p, d in zip(top_payoffs, top_deltas)]

    # Check for intersection
    # Start with the first interval and compare with others
    intersection = intervals[0]
    for interval in intervals[1:]:
        # Update the intersection
        intersection = (max(intersection[0], interval[0]), min(intersection[1], interval[1]))
        # If intervals do not overlap, intersection is empty
        if intersection[0] > intersection[1]:
            return False

    return True

def compact_dictionaries(dict1, dict2):
    """
    Compacts both dictionaries by removing keys where the value of a key 
    is the same as the value of its previous key in both dictionaries,
    for their common keys. Then compacts each dictionary separately.
    """
    # Find common keys
    common_keys = sorted(set(dict1.keys()) & set(dict2.keys()))

    keys_to_remove = []
    prev_value1, prev_value2 = None, None

    for key in common_keys:
        if dict1[key] == prev_value1 and dict2[key] == prev_value2:
            keys_to_remove.append(key)
        prev_value1, prev_value2 = dict1[key], dict2[key]

    for key in keys_to_remove:
        del dict1[key]
        del dict2[key]

    # Compact each dictionary by reassigning keys
    dict1_compacted = {i: dict1[k] for i, k in enumerate(sorted(dict1.keys()))}
    dict2_compacted = {i: dict2[k] for i, k in enumerate(sorted(dict2.keys()))}

    return dict1_compacted, dict2_compacted


def find_unique_rows(matrix, max_num = None ,threshold=0.001):
    unique_rows = []
    unique_indices = dict()
    eff_inx = 0
    unique_indices[0] = eff_inx
    eff_inx += 1

    for i, row in enumerate(matrix):
        if np.sum(row)>0.5:
            if not unique_rows:
                unique_rows.append(row)
                unique_indices[i] = eff_inx
                eff_inx += 1
            else:
                is_unique = True
                for e_i, unique_row in enumerate(unique_rows):
                    if np.linalg.norm(row - unique_row) < threshold:
                        is_unique = False
                        unique_indices[i] = e_i + 1
                        break

                if is_unique:
                    unique_rows.append(row)
                    unique_indices[i] = eff_inx
                    eff_inx += 1
        
        if max_num is not None:
            if eff_inx >= max_num:
                break

    return np.array(unique_rows), unique_indices

class PSRO_learning:
    #   PSRO algorithm

    def __init__(self, args, anchor_policies, shared_policies, eval_policies, runners, eval_envs, role_names, save_dir):
        # Store the input arguments and parameters
        self.args = args
        self.policies_anchor = anchor_policies
        self.policies_shared = shared_policies
        self.eval_policies = eval_policies
        self.runners = runners
        self.eval_envs = eval_envs
        self.role_names = role_names
        self.save_dir = save_dir
        self.n_threads = self.args.n_rollout_threads
        self.n_eval_eps = self.args.eval_episode_num
        self.g_step = 0
        self.total_round = self.args.total_round
        self.use_wandb = self.args.use_wandb
        self.num_env_steps = self.args.num_env_steps
        self.eval = eval_match(self.policies_anchor, self.policies_shared, self.eval_envs)
        self.policy_num = args.population_size
        self.p1_space = [role_names[0] + str(i) for i in range(self.policy_num)]
        self.p2_space = [role_names[1] + str(i) for i in range(self.policy_num)]
        self.graph_generator = prob_matrix(self.p1_space, self.p2_space)
        self.eval_policies[0][0] = self.policies_anchor[0]
        self.eval_policies[1][0] = self.policies_anchor[1]
        self.effect_p2_ids = np.zeros((1,self.policy_num))
        self.effect_p2_ids[0][0] = 1
        self.effect_p1_ids = self.effect_p2_ids
        self.effect_p2_id_inx = [1]
        self.effect_p1_id_inx = [1]
        self.effect_p2_id_map = {0:0}
        self.effect_p1_id_map = {0:0}
        self.waiting_sync = False
        self.use_empty_policy = args.use_empty_policy
        self.until_flat = args.until_flat
        if self.until_flat:
            self.frozen_top_N = args.frozen_top_N
        else:
            self.sub_round = args.sub_round

        if self.use_empty_policy:
            if len(runners[0]) < self.policy_num - 1 or len(runners[1]) < self.policy_num - 1:
                print("The number of runners is less than the number of effective strategies!")
                raise NotImplementedError
        
        self.frozen_p1_inx = 1
        self.frozen_p2_inx = 1
        self.frozen_round = 1
        self.num_f_p1 = None
        self.num_f_p2 = None
        self.frozen_p1_available = True
        self.frozen_p2_available = True
        self.terminal = False

    # Calculate effective policies for the current game state.
    def calu_effective_policy(self):

        max_p1_num = min(self.frozen_p1_inx +1 , self.policy_num)
        max_p2_num = min(self.frozen_p2_inx +1 , self.policy_num)

        # Calculate effective policies for player 1 and player 2 based on probability matrices.
        effect_p1_probs, effect_p1_map = find_unique_rows(matrix=self.graph_generator.p1_prob_mat, max_num=max_p2_num)
        effect_p2_probs, effect_p2_map = find_unique_rows(matrix=self.graph_generator.p2_prob_mat, max_num=max_p1_num)

        max_p1_num = min(self.frozen_p1_inx, effect_p2_probs.shape[0], self.policy_num -1)
        max_p2_num = min(self.frozen_p2_inx, effect_p1_probs.shape[0], self.policy_num -1)


        self.effect_p2_ids = effect_p1_probs[self.frozen_p2_inx-1:max_p2_num, :]
        self.effect_p1_ids = effect_p2_probs[self.frozen_p1_inx-1:max_p1_num, :]

        self.effect_p2_id_inx = np.arange(self.frozen_p2_inx, len(self.effect_p2_ids) + self.frozen_p2_inx,1)
        self.effect_p1_id_inx = np.arange(self.frozen_p1_inx, len(self.effect_p1_ids) + self.frozen_p1_inx,1)


        self.effect_p2_id_map, self.effect_p1_id_map = compact_dictionaries(effect_p1_map, effect_p2_map)

        # Generate indices for effective policies.

        # Print the effective policy IDs for player 1 and player 2.
        print("player1's effective policy id = ", self.effect_p1_ids)
        print("player2's effective policy id = ", self.effect_p2_ids)

        print("player1's effect inx =", self.effect_p1_id_inx)
        print("player2's effect inx =", self.effect_p2_id_inx)

        print("player1's policy inx list = ", self.effect_p1_id_map)
        print("player2's policy inx list = ", self.effect_p2_id_map)

    # Run the training process for the given number of rounds.
    def run(self, available_train_role = None, begin_inx = 1):
        self.warmup(available_train_role=available_train_role)
        print("Training start!")
        for i in range(self.total_round):
            # Perform evaluation for the current round and calculate the prob. matrix
            self.eval_round_step(begin_inx=begin_inx)
            self.cal_graph(round_num=begin_inx)
            if self.terminal:
                self.terminal = False
                break
            begin_inx += 1
            train_done = False
            # Continue training until it is effective
            while train_done == False:
                self.step_run(available_train_role=available_train_role)
                train_done = self.check_training_effectiveness(available_train_role=available_train_role) or True
                print("train_done = ",train_done)
            
        self.eval_round_step(begin_inx=begin_inx)
        self.cal_graph(round_num=self.total_round)

        print("Terminal!")

    # Run a single training round
    def run_single_round(self, begin_inx = 1, available_train_role = None):
        self.eval_round_step(begin_inx=begin_inx)
        self.cal_graph(round_num=begin_inx)
        if self.terminal == False:
            self.step_run(available_train_role=available_train_role)
            train_done = self.check_training_effectiveness(available_train_role=available_train_role)


    # Perform a training step for the specified available training roles.
    def step_run(self, available_train_role = None):
        if available_train_role is None:
            available_train_role = ['player1', 'player2']

        if self.use_empty_policy:
            runner_inx_p1 = self.frozen_p1_inx - 1
            runner_inx_p2 = self.frozen_p2_inx - 1
        else:
            runner_inx_p1 = 0
            runner_inx_p2 = 0

        if 'player1' in available_train_role and self.frozen_p1_available:
            self.runners[0][runner_inx_p1].all_args.global_steps = self.g_step
            self.runners[0][runner_inx_p1].envs.world.oppo_policy = self.mix_policy_p2
            self.runners[0][runner_inx_p1].set_policy_inx(self.frozen_p1_inx)
            self.runners[0][runner_inx_p1].run()
            self.g_step += self.num_env_steps

        if 'player2' in available_train_role and self.frozen_p2_available:
            self.runners[1][runner_inx_p2].all_args.global_steps = self.g_step
            self.runners[1][runner_inx_p2].envs.world.oppo_policy = self.mix_policy_p1
            self.runners[1][runner_inx_p2].set_policy_inx(self.frozen_p2_inx)
            self.runners[1][runner_inx_p2].run()
            self.g_step += self.num_env_steps

    # Check the effectiveness of the training
    def check_training_effectiveness(self, coe_k = 3 ,available_train_role = None):
        
        train_effective = True
        if available_train_role is None:
            available_train_role = ['player1', 'player2']

        if self.use_empty_policy:
            runner_inx_p1 = self.frozen_p1_inx - 1
            runner_inx_p2 = self.frozen_p2_inx - 1
        else:
            runner_inx_p1 = 0
            runner_inx_p2 = 0

        if 'player1' in available_train_role and self.frozen_p1_available:
            eval_values, standard_vaule = self.runners[0][runner_inx_p1].get_payoff_sigma(self.n_eval_eps)
            print("eval_player1 = ", eval_values)
            for inx in range(len(self.effect_p1_id_inx)):
                prob_vector = self.effect_p1_ids[inx]
                #  calculate the payoff before the training
                last_eval_vaule = np.dot(self.payoff_mat[self.effect_p1_id_inx[inx]], prob_vector)

                print("policy {} last_eval_vaule ={}".format(self.effect_p1_id_inx[inx],last_eval_vaule))
                delta_vaule = eval_values - last_eval_vaule
                print("threshold = ", coe_k*standard_vaule)
                sub_done = (delta_vaule > -coe_k*standard_vaule)
                if self.effect_p1_id_inx[inx] == self.frozen_p1_inx:
                    if self.num_f_p1 is None:
                        self.frozen_payoff_p1_buffer = np.array([eval_values])
                        self.frozen_delta_p1_buffer = np.array([standard_vaule])
                    else:
                        self.frozen_payoff_p1_buffer = np.append(self.frozen_payoff_p1_buffer, eval_values)
                        self.frozen_delta_p1_buffer = np.append(self.frozen_delta_p1_buffer, standard_vaule)
                    
                    self.num_f_p1 = len(self.frozen_payoff_p1_buffer) - 1

                    print("frozen_p1_num = ", self.num_f_p1)

                    self.runners[0][runner_inx_p1].save_as_filename("frozen_" + str(self.num_f_p1) + "_backup")
                    np.save(os.path.join(self.save_dir, 'frozen_delta_buffer_'+str(self.role_names[0])+'.npy'), np.array(self.frozen_delta_p1_buffer))
                    np.save(os.path.join(self.save_dir, 'frozen_payoff_buffer_'+str(self.role_names[0])+'.npy'), np.array(self.frozen_payoff_p1_buffer))
                    print("{}_{} policy backup {}".format(self.role_names[0],self.frozen_p1_inx,self.num_f_p1))
                            
                train_effective = train_effective and sub_done

        if 'player2' in available_train_role and self.frozen_p2_available:
            eval_values, standard_vaule = self.runners[1][runner_inx_p2].get_payoff_sigma(self.n_eval_eps)
            print("eval_player2 = ", eval_values)
            for inx in range(len(self.effect_p2_id_inx)):
                prob_vector = self.effect_p2_ids[inx]
                #  calculate the payoff before the training
                last_eval_vaule = np.dot(-self.payoff_mat[:, self.effect_p2_id_inx[inx]], prob_vector)

                print("policy {} last_eval_vaule ={}".format(self.effect_p2_id_inx[inx],last_eval_vaule))
                delta_vaule = (eval_values - last_eval_vaule)
                print("threshold = ", coe_k*standard_vaule)
                sub_done = (delta_vaule > -coe_k*standard_vaule)
                if self.effect_p2_id_inx[inx] == self.frozen_p2_inx:
                    if self.num_f_p2 is None:
                        self.frozen_payoff_p2_buffer = np.array([eval_values])
                        self.frozen_delta_p2_buffer = np.array([standard_vaule])
                    else:
                        self.frozen_payoff_p2_buffer = np.append(self.frozen_payoff_p2_buffer, eval_values)
                        self.frozen_delta_p2_buffer = np.append(self.frozen_delta_p2_buffer, standard_vaule)

                    self.num_f_p2 = len(self.frozen_payoff_p2_buffer) - 1

                    print("frozen_p2_num = ", self.num_f_p2)
                        
                    self.runners[1][runner_inx_p2].save_as_filename("frozen_" + str(self.num_f_p2) + "_backup")
                    np.save(os.path.join(self.save_dir, 'frozen_delta_buffer_'+str(self.role_names[1])+'.npy'), np.array(self.frozen_delta_p2_buffer))
                    np.save(os.path.join(self.save_dir, 'frozen_payoff_buffer_'+str(self.role_names[1])+'.npy'), np.array(self.frozen_payoff_p2_buffer))
                    print("{}_{} policy backup {}".format(self.role_names[1],self.frozen_p2_inx,self.num_f_p2))

                train_effective = train_effective and sub_done

        return train_effective

    # Perform evaluation steps before a training round
    def eval_round_step(self, begin_inx):
        # cover_inx = max(1, min(begin_inx, self.policy_num))
        if begin_inx <= 1:
            eval_range = np.array([1, 1])
            self.frozen_id_p1 = None
            self.frozen_id_p2 = None
            self.frozen_payoff_mat = None
        else:
            eval_range = np.array([int(self.frozen_p1_available) + self.frozen_p1_inx, int(self.frozen_p2_available) + self.frozen_p2_inx])
            if self.frozen_p1_available:
                self.frozen_payoff_p1_buffer = np.load(os.path.join(self.save_dir, 'frozen_payoff_buffer_'+str(self.role_names[0])+'.npy'))
                self.frozen_delta_p1_buffer = np.load(os.path.join(self.save_dir, 'frozen_delta_buffer_'+str(self.role_names[0])+'.npy'))
                self.num_f_p1 = len(self.frozen_payoff_p1_buffer) - 1
            if self.frozen_p2_available:
                self.frozen_payoff_p2_buffer = np.load(os.path.join(self.save_dir, 'frozen_payoff_buffer_'+str(self.role_names[1])+'.npy'))
                self.frozen_delta_p2_buffer = np.load(os.path.join(self.save_dir, 'frozen_delta_buffer_'+str(self.role_names[1])+'.npy'))
                self.num_f_p2 = len(self.frozen_payoff_p2_buffer) - 1
            if self.frozen_p1_inx >1:
                self.frozen_id_p1 = np.load(os.path.join(self.save_dir, 'frozen_id_'+str(self.role_names[0])+'.npy'))
            else:
                self.frozen_id_p1 = None
            if self.frozen_p2_inx >1:
                self.frozen_id_p2 = np.load(os.path.join(self.save_dir, 'frozen_id_'+str(self.role_names[1])+'.npy'))
            else:
                self.frozen_id_p2 = None
            self.frozen_payoff_mat = np.load(os.path.join(self.save_dir, 'frozen_payoff_mat.npy'))

            if self.use_empty_policy == False:
                self.runners[0][0].inherit_policy(self.save_dir, str(self.role_names[0]))
                self.runners[1][0].inherit_policy(self.save_dir, str(self.role_names[1]))
        
        # Restore evaluation policies of player1 and player2
        for j in range(1, eval_range[0]):
            if j < self.frozen_p1_inx:
                restore_eval_policy(self.eval_policies[0][j], self.save_dir, self.role_names[0], self.policies_shared[0].use_mixer ,head_str=("frozen_policy_" + str(j)))
            elif j == self.frozen_p1_inx:
                if self.frozen_p1_available:
                    if self.use_empty_policy:
                        policy_inx = len(self.frozen_payoff_p1_buffer) - 1
                    else:
                        policy_inx = np.argmax(self.frozen_payoff_p1_buffer)
                    restore_eval_policy(self.eval_policies[0][j], self.save_dir, self.role_names[0], self.policies_shared[0].use_mixer ,head_str=("frozen_" + str(policy_inx) + "_backup"))
                else:
                    restore_eval_policy(self.eval_policies[0][j], self.save_dir, self.role_names[0], self.policies_shared[0].use_mixer)
            else:
                restore_eval_policy(self.eval_policies[0][j], self.save_dir, self.role_names[0], self.policies_shared[0].use_mixer)
            
        for j in range(1, eval_range[1]):
            if j < self.frozen_p2_inx:
                restore_eval_policy(self.eval_policies[1][j], self.save_dir, self.role_names[1], self.policies_shared[1].use_mixer, head_str=("frozen_policy_" + str(j)))
            elif j == self.frozen_p2_inx:
                if self.frozen_p2_available:
                    if self.use_empty_policy:
                        policy_inx = len(self.frozen_payoff_p2_buffer) - 1
                    else:
                        policy_inx = np.argmax(self.frozen_payoff_p2_buffer)
                    restore_eval_policy(self.eval_policies[1][j], self.save_dir, self.role_names[1], self.policies_shared[1].use_mixer, head_str=("frozen_" + str(policy_inx) + "_backup"))
                else:
                    restore_eval_policy(self.eval_policies[1][j], self.save_dir, self.role_names[1], self.policies_shared[1].use_mixer)
            else:
                restore_eval_policy(self.eval_policies[1][j], self.save_dir, self.role_names[1], self.policies_shared[1].use_mixer)

        self.eval = eval_match(self.eval_policies[0][0:eval_range[0]], self.eval_policies[1][0:eval_range[1]], self.eval_envs)

        # Calculate win prob. matrices and payoff matrix
        if begin_inx <= 1:
            self.eval.get_win_prob_mat(self.n_threads, self.n_eval_eps, gamma=self.args.gamma)
        else:
            sub_inx_1 = [self.frozen_p1_inx, eval_range[0], 0, eval_range[1]]
            sub_inx_2 = [0, self.frozen_p1_inx, self.frozen_p2_inx, eval_range[1]]
            self.eval.get_win_prob_mat(self.n_threads, self.n_eval_eps, sub_mat_inx= sub_inx_1, gamma=self.args.gamma)
            self.eval.get_win_prob_mat(self.n_threads, self.n_eval_eps, sub_mat_inx= sub_inx_2, gamma=self.args.gamma)
        # payoff_mat = self.eval.total_round_mat * (2 * self.eval.win_prob_mat -1)/(self.n_threads*self.n_eval_eps)
        mask_positive = self.eval.total_round_mat > 0.5
        payoff_mat = np.zeros_like(self.eval.total_round_mat)
        payoff_mat[mask_positive] = (self.eval.total_reward_mat[mask_positive])/(self.eval.total_round_mat[mask_positive])
        # payoff_mat[0:self.frozen_payoff_mat.shape[0]][0:self.frozen_payoff_mat.shape[1]] = copy.deepcopy(self.frozen_payoff_mat)

        # Save the payoff matrix as a file for multi-machine parallel training
        np.save(os.path.join(self.save_dir, 'payoff_sync.npy'), payoff_mat)

    # Calculate and update the game graph based on evaluation results
    def cal_graph(self, round_num, mat_sync = False):
        # cover_inx = max(1, min(begin_inx, self.policy_num -1)) + 1
        dict_mat_p1 = dict()
        dict_mat_p2 = dict()

        # Check matrix sync to ensure the same payoff matrix is used in all machines
        if mat_sync:
            prob_mat_sync = np.load(os.path.join(self.save_dir, 'payoff_sync.npy'))
            mask_positive = self.eval.total_round_mat > 0.5
            payoff_mat = np.zeros_like(self.eval.total_round_mat)
            payoff_mat[mask_positive] = (self.eval.total_reward_mat[mask_positive])/(self.eval.total_round_mat[mask_positive])
            payoff_mat = (payoff_mat + prob_mat_sync)/2
        else:
            mask_positive = self.eval.total_round_mat > 0.5
            payoff_mat = np.zeros_like(self.eval.total_round_mat)
            payoff_mat[mask_positive] = (self.eval.total_reward_mat[mask_positive])/(self.eval.total_round_mat[mask_positive])

        if self.frozen_payoff_mat is not None:
            payoff_mat[0:self.frozen_p1_inx, 0:self.frozen_p2_inx] = copy.deepcopy(self.frozen_payoff_mat)

        # payoff_mat = np.pad(payoff_mat, ((0, self.policy_num - payoff_mat.shape[0]), (0, self.policy_num - payoff_mat.shape[1])), 'constant', constant_values=0)
        self.graph_generator.update_prob_matrix(payoff_mat,self.effect_p1_id_map,self.effect_p2_id_map)
        dict_mat_p1["round_" + str(round_num) + "_win_prob_mat"] = self.eval.win_prob_mat
        dict_mat_p1["round_" + str(round_num) + "_probs_p1_mat"] = self.graph_generator.p1_prob_mat
        dict_mat_p1["round_" + str(round_num) + "_payoff_p1_mat"] = payoff_mat
        dict_mat_p2["round_" + str(round_num) + "_probs_p2_mat"] = self.graph_generator.p2_prob_mat

        if round_num > 1:
            if self.until_flat:
                if self.num_f_p1 >=self.frozen_top_N - 1 and self.frozen_p1_available:
                    done_player1 = check_convergence(self.frozen_payoff_p1_buffer, self.frozen_delta_p1_buffer, coeff_std=3, N=self.frozen_top_N)
                else:
                    done_player1 = not self.frozen_p1_available
            else:
                if self.num_f_p1 >= self.sub_round - 1 and self.frozen_p1_available:
                    done_player1 = True
                else:
                    done_player1 = not self.frozen_p1_available

            if self.until_flat:
                if self.num_f_p2 >=self.frozen_top_N - 1 and self.frozen_p2_available:
                    done_player2 = check_convergence(self.frozen_payoff_p2_buffer, self.frozen_delta_p2_buffer, coeff_std=3, N=self.frozen_top_N)
                else:
                    done_player2 = not self.frozen_p2_available
            else:
                if self.num_f_p2 >= self.sub_round - 1 and self.frozen_p2_available:
                    done_player2 = True
                else:
                    done_player2 = not self.frozen_p2_available
            
            print("payoff_line_p1 = ")
            print(self.frozen_payoff_p1_buffer)
            print("payoff_line_p2 = ")
            print(self.frozen_payoff_p2_buffer)
        else:
            done_player1 = False
            done_player2 = False


        if done_player1 and done_player2:
            self.frozen_round += 1
            self.num_f_p1 = None
            self.num_f_p2 = None

            frozen_p1_available_nxt = True
            frozen_p2_available_nxt = True
        
            nxt_p1_id = self.graph_generator.p2_prob_mat[self.frozen_round]
            nxt_p2_id = self.graph_generator.p1_prob_mat[self.frozen_round]

            if self.frozen_p1_available:
                print(f"frozen '{self.role_names[0]}' policy '{self.frozen_p1_inx}'!")
                max_inx = np.argmax(self.frozen_payoff_p1_buffer)
                last_str = 'frozen_' + str(max_inx) + '_backup_' + self.role_names[0]
                new_str = 'frozen_policy_' + str(self.frozen_p1_inx) + '_' + self.role_names[0]
                os.rename(os.path.join(self.save_dir, "actor_" + last_str + ".pt"), os.path.join(self.save_dir, "actor_" + new_str + ".pt"))
                print(f"File has been renamed from '{last_str}.pt' to '{new_str}.pt'.")
                # self.frozen_p1_inx += 1
                nxt_frozen_p1_inx = self.frozen_p1_inx + 1
                if self.frozen_id_p1 is None:
                    nxt_frozen_p1_id = np.expand_dims(self.effect_p1_ids[0], axis=0)
                else:
                    nxt_frozen_p1_id = np.pad(copy.deepcopy(self.frozen_id_p1), ((0, 1), (0, 0)), 'constant', constant_values=0)
                    nxt_frozen_p1_id[-1] = self.effect_p1_ids[0]
                    # self.frozen_id_p1 = nxt_frozen_p1_id
            else:
                nxt_frozen_p1_inx = self.frozen_p1_inx
                nxt_frozen_p1_id = self.frozen_id_p1
            if self.frozen_p2_available:
                print(f"frozen '{self.role_names[1]}' policy '{self.frozen_p2_inx}'!")
                max_inx = np.argmax(self.frozen_payoff_p2_buffer)
                last_str = 'frozen_' + str(max_inx) + '_backup_' + self.role_names[1]
                new_str = 'frozen_policy_' + str(self.frozen_p2_inx) + '_' + self.role_names[1]
                os.rename(os.path.join(self.save_dir, "actor_" + last_str + ".pt"), os.path.join(self.save_dir, "actor_" + new_str + ".pt"))
                print(f"File has been renamed from '{last_str}.pt' to '{new_str}.pt'.")
                # self.frozen_p2_inx += 1
                nxt_frozen_p2_inx = self.frozen_p2_inx + 1
                if self.frozen_id_p2 is None:
                    nxt_frozen_p2_id = np.expand_dims(self.effect_p2_ids[0], axis=0)
                else:
                    nxt_frozen_p2_id = np.pad(copy.deepcopy(self.frozen_id_p2), ((0, 1), (0, 0)), 'constant', constant_values=0)
                    nxt_frozen_p2_id[-1] = self.effect_p2_ids[0]
                    # self.frozen_id_p2 = nxt_frozen_p2_id
            else:
                nxt_frozen_p2_inx = self.frozen_p2_inx
                nxt_frozen_p2_id = self.frozen_id_p2

            print("nxt_frozen_p1_inx = ", nxt_frozen_p1_inx)
            print("nxt_frozen_p2_inx = ", nxt_frozen_p2_inx)


            for p in range(1, nxt_frozen_p1_inx):
                if np.linalg.norm(nxt_p1_id - nxt_frozen_p1_id[p-1]) < 0.0001:
                    frozen_p1_available_nxt = False
                    break
            for p in range(1, nxt_frozen_p2_inx):
                if np.linalg.norm(nxt_p2_id - nxt_frozen_p2_id[p-1]) < 0.0001:
                    frozen_p2_available_nxt = False
                    break

            # print(nxt_p1_id, nxt_frozen_p1_id)
            # print(nxt_p1_id, nxt_frozen_p2_id)
            
            print("frozen_p1_available_nxt = ",frozen_p1_available_nxt)
            print("frozen_p2_available_nxt = ", frozen_p2_available_nxt)

            if frozen_p1_available_nxt or frozen_p2_available_nxt:
                print("new frozen policy!")
                self.frozen_id_p1 = nxt_frozen_p1_id
                self.frozen_id_p2 = nxt_frozen_p2_id
                self.frozen_p1_inx = nxt_frozen_p1_inx
                self.frozen_p2_inx = nxt_frozen_p2_inx

                if self.frozen_p1_inx >= self.policy_num or self.frozen_p2_inx >= self.policy_num:
                    self.terminal = True
        
                print("save frozen payoff mat and id.")
                np.save(os.path.join(self.save_dir, 'frozen_id_'+str(self.role_names[0])+'.npy'), self.frozen_id_p1)
                np.save(os.path.join(self.save_dir, 'frozen_id_'+str(self.role_names[1])+'.npy'), self.frozen_id_p2)
                
                self.frozen_p1_available = frozen_p1_available_nxt
                self.frozen_p2_available = frozen_p2_available_nxt

                para_dict = {
                    "frozen_p1_inx": self.frozen_p1_inx,
                    "frozen_p2_inx": self.frozen_p2_inx,
                    "frozen_p1_available": self.frozen_p1_available,
                    "frozen_p2_available": self.frozen_p2_available,
                    "frozen_round": self.frozen_round
                }
                print("new frozen parameter:")
                print(para_dict)

                with open(os.path.join(self.save_dir, 'frozen_para.json'), 'w') as f:
                    json.dump(para_dict, f, indent=4)
            else:
                self.frozen_round -= 1
                print("Pseudo-convergence!")

            pattern = re.compile(r'frozen_(\d+)_backup')

            for filename in os.listdir(self.save_dir):
                if pattern.search(filename):
                    file_path = os.path.join(self.save_dir, filename)
                    os.remove(file_path)
                    print(f"Deleted file: {filename}")

            pattern = re.compile(r'buffer')

            for filename in os.listdir(self.save_dir):
                if pattern.search(filename):
                    file_path = os.path.join(self.save_dir, filename)
                    os.remove(file_path)
                    print(f"Deleted file: {filename}")

    
        new_frozen_payoff_mat = payoff_mat[0:self.frozen_p1_inx, 0:self.frozen_p2_inx]
        np.save(os.path.join(self.save_dir, 'frozen_payoff_mat.npy'), new_frozen_payoff_mat)
            
        payoff_mat_last_col = payoff_mat[:,-1]
        payoff_mat_last_row = payoff_mat[-1]

        self.payoff_mat = np.pad(payoff_mat, ((0, self.policy_num - payoff_mat.shape[0]), (0, self.policy_num - payoff_mat.shape[1])), 'constant', constant_values=0)

        # Extend the payoff matrix for calculating the effectiveness of possible new policies
        if payoff_mat.shape[0] <self.policy_num:
            for i in range(payoff_mat.shape[0], self.policy_num):
                self.payoff_mat[i, 0:payoff_mat.shape[1]] = payoff_mat_last_row

        if payoff_mat.shape[1] <self.policy_num:
            for i in range(payoff_mat.shape[1], self.policy_num):
                self.payoff_mat[0:payoff_mat.shape[0], i] = payoff_mat_last_col

        # Calculate effective policies and save them
        if self.terminal == False:
            self.calu_effective_policy()
        np.save(os.path.join(self.save_dir, 'probs_p1.npy'), self.graph_generator.p1_prob_mat)
        np.save(os.path.join(self.save_dir, 'probs_p2.npy'), self.graph_generator.p2_prob_mat)

        print("prob_p1=")
        print(self.graph_generator.p1_prob_mat)
        print("prob_p2=")
        print(self.graph_generator.p2_prob_mat)

        # Create mixing policies for player 1 and player 2
        if self.terminal == False:
            self.mix_policy_p1 = mixing_policy(self.n_threads, self.eval_policies[0], probs=self.effect_p2_ids[0,:])
            self.mix_policy_p2 = mixing_policy(self.n_threads, self.eval_policies[1], probs=self.effect_p1_ids[0,:])


        if self.use_wandb:
            for k, v in dict_mat_p1.items():
                wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p2_space)})
            for k, v in dict_mat_p2.items():
                wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p1_space)})
            wandb.log({"train_rounds": round_num})
            wandb.log({"effect_policy_number_p1": len(self.effect_p2_id_inx)})
            wandb.log({"effect_policy_number_p2": len(self.effect_p1_id_inx)})

    # Load effective policy id from saved files
    def load_effect_sigma(self):
        probs_p1 = np.load(os.path.join(self.save_dir, 'probs_p1.npy'))
        probs_p2 = np.load(os.path.join(self.save_dir, 'probs_p2.npy'))
        self.graph_generator.p1_prob_mat = probs_p1
        self.graph_generator.p2_prob_mat = probs_p2

        self.calu_effective_policy()

    def restore(self):
        if os.path.exists(os.path.join(self.save_dir, 'frozen_para.json')):
            with open(os.path.join(self.save_dir, 'frozen_para.json'), 'r') as f:
                frozen_para = json.load(f)
            self.frozen_p1_inx = frozen_para["frozen_p1_inx"]
            self.frozen_p2_inx = frozen_para["frozen_p2_inx"]
            self.frozen_round = frozen_para["frozen_round"]
            self.frozen_p1_available = frozen_para["frozen_p1_available"]
            self.frozen_p2_available = frozen_para["frozen_p2_available"]
            print('load parameters:')
            print(frozen_para)
        else:
            print("File 'frozen_para.json' does not exist.")
        if self.use_empty_policy:
            runner_inx_p1 = self.frozen_p1_inx - 1
            runner_inx_p2 = self.frozen_p2_inx - 1
        else:
            runner_inx_p1 = 0
            runner_inx_p2 = 0
        self.runners[0][runner_inx_p1].inherit_policy(self.save_dir,self.role_names[0])
        self.runners[1][runner_inx_p2].inherit_policy(self.save_dir,self.role_names[1])
        self.load_effect_sigma()

    def isTerminal(self):
        return copy.deepcopy(self.terminal)

    # Perform a warm-up for NeuPL.
    def warmup(self, available_train_role = None):
        print("NeuPL warming up!")
        if available_train_role is None:
            available_train_role = ['player1', 'player2']
        if 'player1' in available_train_role:
            self.runners[0][0].save()
        if 'player2' in available_train_role:
            self.runners[1][0].save()
        

