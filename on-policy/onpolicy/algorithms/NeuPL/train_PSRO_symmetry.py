# TODO:

import numpy as np
import torch
import wandb
from onpolicy.algorithms.NeuPL.Policy_prob_matrix import Nash_matrix as prob_matrix
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as mixing_policy
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as Policy
import random
import os
import re
import time
import copy
import json

def restore_eval_policy(policy, model_dir, use_mixer = True, head_str = None):
    """Restore policy's networks from a saved model."""
    if head_str is None:
        policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
        policy.actor.load_state_dict(policy_actor_state_dict)
        if use_mixer:
            policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer.pt')
            policy.mixer.load_state_dict(policy_mixer_state_dict)
    else:
        policy_actor_state_dict = torch.load(str(model_dir)  + '/actor_' + str(head_str) + '.pt')
        policy.actor.load_state_dict(policy_actor_state_dict)
        if use_mixer:
            policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer_' + str(head_str) + '.pt')
            policy.mixer.load_state_dict(policy_mixer_state_dict)

def creat_policy(args, envs, save_dir, device, use_mixer):
    policy = Policy(args,
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_act_space[0],
                        device)
    restore_eval_policy(policy, save_dir, use_mixer)
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
    #   Neupl is used for a two-player zero-sum game scenario, consisting of a pair of policy population {\pi(a|o,sigma_1), \pi(a|o,sigma_2)}
    #   sigma_{i} represents the ID of player{i}'s policy, and it also signifies the mixed probability of opponent's policy

    """
    list of generation flies:
    effect id
    """

    def __init__(self, args, anchor_policies, empty_policies, eval_policies, runners, evaluator, role_names, save_dir, device = torch.device("cpu")):
        # Store the input arguments and parameters
        # Store the input arguments and parameters
        self.args = args
        self.policies_anchor = anchor_policies
        self.empty_policies = empty_policies
        self.eval_policies = eval_policies
        self.runners = runners
        self.eval = evaluator
        self.role_names = role_names
        self.save_dir = save_dir
        self.device = device
        self.n_threads = self.args.n_rollout_threads
        self.n_eval_eps = self.args.eval_episode_num
        self.g_step = 0
        self.total_round = self.args.total_round
        self.use_wandb = self.args.use_wandb
        self.use_calc_exploit = args.use_calc_exploit
        self.num_env_steps = self.args.num_env_steps
        self.policy_num = self.args.population_size
        self.p1_space = [role_names[0] + str(i) for i in range(self.policy_num)]
        self.graph_generator = prob_matrix(self.p1_space, self.p1_space)
        self.eval_policies[0] = self.policies_anchor[0]
        self.effect_id_map = {0:0}
        self.until_flat = args.until_flat
        self.use_empty_policy = args.use_empty_policy
        if self.until_flat:
            self.frozen_top_N = args.frozen_top_N
        else:
            self.sub_round = args.sub_round

        if self.use_empty_policy:
            if len(runners) < self.policy_num - 1:
                print("No enough runners, the algo needs {} runners, but only {} runners exist now".format(self.policy_num-1, len(runners)))
                raise NotImplementedError
            
            
        self.frozen_inx = 1
        self.frozen_round = 1
        self.num_f = None
        self.terminal = False

    # Calculate effective policies for the current game state.
    def calu_effective_policy(self):
            
        max_num = min(self.frozen_inx + 1 , self.policy_num)

        # Calculate effective policies for player 1 and player 2 based on probability matrices.
        effect_p1_probs, effect_p1_map = find_unique_rows(matrix=self.graph_generator.p1_prob_mat, max_num=max_num)

        max_num = min(self.frozen_inx, effect_p1_probs.shape[0], self.policy_num -1)

        self.effect_ids = effect_p1_probs[self.frozen_inx-1:max_num, :]
        self.effect_id_inx = np.arange(self.frozen_inx, len(self.effect_ids) + self.frozen_inx,1)


        self.effect_id_map, _ = compact_dictionaries(effect_p1_map, copy.deepcopy(effect_p1_map))

        # Generate indices for effective policies.

        # Print the effective policy IDs for player 1 and player 2.
        print("player's effective policy id = ", self.effect_ids)

        print("player's effect inx =", self.effect_id_inx)

        print("player's policy inx list = ", self.effect_id_map)

    # Run the training process for the given number of rounds.
    def run(self, begin_inx = 1):
        self.warmup()
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
                self.step_run()
                train_done = self.check_training_effectiveness() or True
                print("train_done = ",train_done)
            
        self.eval_round_step(begin_inx=begin_inx)
        self.cal_graph(round_num=self.total_round)

        print("Terminal!")

    # Run a single training round
    def run_single_round(self, begin_inx = 1):
        self.eval_round_step(begin_inx=begin_inx)
        self.cal_graph(round_num=begin_inx)
        train_done = False
        while train_done == False:
            self.step_run()
            train_done = self.check_training_effectiveness() or True
            print("train_done = ",train_done)


    # Perform a training step for the specified available training roles.
    def step_run(self):

        if self.use_empty_policy:
            runner_inx = self.frozen_inx - 1
        else:
            runner_inx = 0

        self.runners[runner_inx].all_args.global_steps = self.g_step
        self.runners[runner_inx].envs.world.oppo_policy = self.mix_policy
        self.runners[runner_inx].set_policy_inx(self.frozen_inx)
        self.runners[runner_inx].run()
        self.g_step += self.num_env_steps


    # Check the effectiveness of the training
    def check_training_effectiveness(self, coe_k = 3):
        
        train_effective = True

        if self.use_empty_policy:
            runner_inx = self.frozen_inx - 1
        else:
            runner_inx = 0
        
        eval_values, standard_vaule = self.runners[runner_inx].get_payoff_sigma(self.n_eval_eps)
        print("eval_player1 = ", eval_values)
        for inx in range(len(self.effect_id_inx)):
            prob_vector = self.effect_ids[inx]
            #  calculate the payoff before the training
            last_eval_vaule = np.dot(self.payoff_mat[self.effect_id_inx[inx]], prob_vector)

            print("policy {} last_eval_vaule ={}".format(self.effect_id_inx[inx],last_eval_vaule))
            delta_vaule = eval_values - last_eval_vaule
            print("threshold = ", coe_k*standard_vaule)
            sub_done = (delta_vaule > -coe_k*standard_vaule)
            if self.effect_id_inx[inx] == self.frozen_inx:
                if self.num_f is None:
                    self.frozen_payoff_buffer = np.array([eval_values])
                    self.frozen_delta_buffer = np.array([standard_vaule])
                else:
                    self.frozen_payoff_buffer = np.append(self.frozen_payoff_buffer, eval_values)
                    self.frozen_delta_buffer = np.append(self.frozen_delta_buffer, standard_vaule)
                
                self.num_f = len(self.frozen_payoff_buffer) - 1

                print("frozen_num = ", self.num_f)

                self.runners[runner_inx].save_as_filename("frozen_" + str(self.num_f) + "_backup")
                np.save(os.path.join(self.save_dir, 'frozen_delta_buffer_'+str(self.role_names[0])+'.npy'), np.array(self.frozen_delta_buffer))
                np.save(os.path.join(self.save_dir, 'frozen_payoff_buffer_'+str(self.role_names[0])+'.npy'), np.array(self.frozen_payoff_buffer))
                print("{}_{} policy backup {}".format(self.role_names[0],self.frozen_inx,self.num_f))
                        
            train_effective = train_effective and sub_done


        return train_effective

    # Perform evaluation steps before a training round
    def eval_round_step(self, begin_inx):
        # cover_inx = max(1, min(begin_inx, self.policy_num))
        if begin_inx <= 1:
            eval_range = 1
            self.frozen_id = None
            self.frozen_payoff_mat = None
        else:
            eval_range =len(self.effect_id_inx) + self.frozen_inx
            self.frozen_payoff_buffer = np.load(os.path.join(self.save_dir, 'frozen_payoff_buffer_'+str(self.role_names[0])+'.npy'))
            self.frozen_delta_buffer = np.load(os.path.join(self.save_dir, 'frozen_delta_buffer_'+str(self.role_names[0])+'.npy'))
            self.num_f = len(self.frozen_payoff_buffer) - 1
            
            if self.frozen_inx >1:
                self.frozen_id = np.load(os.path.join(self.save_dir, 'frozen_id_'+str(self.role_names[0])+'.npy'))
            else:
                self.frozen_id = None
            self.frozen_payoff_mat = np.load(os.path.join(self.save_dir, 'frozen_payoff_mat.npy'))
            
            if self.use_empty_policy:
                runner_inx = self.frozen_inx - 1
            else:
                runner_inx = 0
                self.runners[runner_inx].inherit_policy(self.save_dir)



        
        # Restore evaluation policies of player1 and player2
        for j in range(1, eval_range):
            if j < self.frozen_inx:
                restore_eval_policy(self.eval_policies[j], self.save_dir, self.empty_policies[0].use_mixer ,head_str=("frozen_policy_" + str(j)))
            elif j == self.frozen_inx:
                if self.until_flat:
                    max_inx = np.argmax(self.frozen_payoff_buffer)
                    restore_eval_policy(self.eval_policies[j], self.save_dir, self.empty_policies[0].use_mixer ,head_str=("frozen_" + str(max_inx) + "_backup"))
                else:
                    last_inx = len(self.frozen_payoff_buffer) - 1
                    restore_eval_policy(self.eval_policies[j], self.save_dir, self.empty_policies[0].use_mixer ,head_str=("frozen_" + str(last_inx) + "_backup"))
            else:
                restore_eval_policy(self.eval_policies[j], self.save_dir, self.empty_policies[0].use_mixer)

        self.eval.update_policy(self.eval_policies[0:eval_range])

        # Calculate win prob. matrices and payoff matrix
        if begin_inx <= 1:
            self.eval.get_win_prob_mat(self.n_threads, self.n_eval_eps)
            self.payoff_mat_ = np.zeros((1,1))
        else:
            self.mask_eval = np.zeros((eval_range, eval_range), dtype=bool)
            for p in range(self.frozen_inx, eval_range):
                for q in range(p):
                    self.mask_eval[p][q] = True
            self.payoff_mat_ = self.eval.get_win_prob_with_mask(self.n_threads, self.n_eval_eps, mask=self.mask_eval)

    # Calculate and update the game graph based on evaluation results
    def cal_graph(self, round_num):
        # cover_inx = max(1, min(begin_inx, self.policy_num -1)) + 1
        dict_mat_p1 = dict()
        dict_mat_p2 = dict()

        # payoff_mat = self.eval.total_round_mat * (2 * self.eval.win_prob_mat -1)/(self.n_threads*self.n_eval_eps)
        payoff_mat_ = copy.deepcopy(self.payoff_mat_)
        payoff_mat = payoff_mat_ - payoff_mat_.T
        # payoff_mat[0:self.frozen_payoff_mat.shape[0]][0:self.frozen_payoff_mat.shape[1]] = copy.deepcopy(self.frozen_payoff_mat)

        if self.frozen_payoff_mat is not None:
            payoff_mat[0:self.frozen_inx, 0:self.frozen_inx] = copy.deepcopy(self.frozen_payoff_mat)

        # payoff_mat = np.pad(payoff_mat, ((0, self.policy_num - payoff_mat.shape[0]), (0, self.policy_num - payoff_mat.shape[1])), 'constant', constant_values=0)
        self.graph_generator.update_prob_matrix(payoff_mat,self.effect_id_map,self.effect_id_map)
        if self.use_calc_exploit:
            probs, _ = self.graph_generator.caul_prob_from_payoff(payoff_mat)
            self.nash_probs = np.array(probs[0])
        # dict_mat_p1["round_" + str(round_num) + "_win_prob_mat"] = self.eval.win_prob_mat
        dict_mat_p1["round_" + str(round_num) + "_probs_p1_mat"] = self.graph_generator.p1_prob_mat
        dict_mat_p1["round_" + str(round_num) + "_payoff_p1_mat"] = payoff_mat
        dict_mat_p2["round_" + str(round_num) + "_probs_p2_mat"] = self.graph_generator.p2_prob_mat

        if round_num > 1:
            if self.until_flat:
                if self.num_f >=self.frozen_top_N - 1:
                    done_player = check_convergence(self.frozen_payoff_buffer, self.frozen_delta_buffer, coeff_std=3, N=self.frozen_top_N)
                else:
                    done_player = False
            else:
                if self.num_f >=self.sub_round - 1:
                    done_player = True
                else:
                    done_player = False

            print("payoff_line = ")
            print(self.frozen_payoff_buffer)
        else:
            done_player = False


        if done_player:
            self.frozen_round += 1
            self.num_f = None
            
            print(f"frozen '{self.role_names[0]}' policy '{self.frozen_inx}'!")
            max_inx = np.argmax(self.frozen_payoff_buffer)
            last_str = 'frozen_' + str(max_inx) + '_backup'
            new_str = 'frozen_policy_' + str(self.frozen_inx)
            os.rename(os.path.join(self.save_dir, "actor_" + last_str + ".pt"), os.path.join(self.save_dir, "actor_" + new_str + ".pt"))
            print(f"File has been renamed from '{last_str}.pt' to '{new_str}.pt'.")
            # self.frozen_p1_inx += 1
            nxt_frozen_inx = self.frozen_inx + 1
            if self.frozen_id is None:
                nxt_frozen_id = np.expand_dims(self.effect_ids[0], axis=0)
            else:
                nxt_frozen_id = np.pad(copy.deepcopy(self.frozen_id), ((0, 1), (0, 0)), 'constant', constant_values=0)
                nxt_frozen_id[-1] = self.effect_ids[0]
                # self.frozen_id_p1 = nxt_frozen_p1_id
            

            print("nxt_frozen_inx = ", nxt_frozen_inx)
            nxt_id = self.graph_generator.p2_prob_mat[self.frozen_round]

            frozen_available_nxt = True
            for p in range(1, nxt_frozen_inx):
                if np.linalg.norm(nxt_id - nxt_frozen_id[p-1]) < 0.0001:
                    frozen_available_nxt = False
                    break

            # print(nxt_p1_id, nxt_frozen_p1_id)
            # print(nxt_p1_id, nxt_frozen_p2_id)
            
            print("frozen_available_nxt = ",frozen_available_nxt)

            if frozen_available_nxt:
                print("new frozen policy!")
                self.frozen_id = nxt_frozen_id
                self.frozen_inx = nxt_frozen_inx

                if self.frozen_inx >= self.policy_num:
                    self.terminal = True
        
                print("save frozen payoff mat and id.")
                np.save(os.path.join(self.save_dir, 'frozen_id_'+str(self.role_names[0])+'.npy'), self.frozen_id)

                para_dict = {
                    "frozen_inx": self.frozen_inx,
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

        new_frozen_payoff_mat = payoff_mat[0:self.frozen_inx, 0:self.frozen_inx]
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
            self.mix_policy = mixing_policy(self.n_threads, self.eval_policies, probs= self.effect_ids[0, :], device=self.device)


        if self.use_wandb:
            for k, v in dict_mat_p1.items():
                wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p1_space)})
            for k, v in dict_mat_p2.items():
                wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p1_space)})
            wandb.log({"train_rounds": round_num})
            wandb.log({"effect_policy_number": len(self.effect_id_inx)})

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
            self.frozen_inx = frozen_para["frozen_inx"]
            self.frozen_round = frozen_para["frozen_round"]
            print('load parameters:')
            print(frozen_para)
        else:
            print("File 'frozen_para.json' does not exist.")
        self.load_effect_sigma()

    def isTerminal(self):
        return copy.deepcopy(self.terminal)
    
    def get_sub_nash_policy(self):
        return copy.deepcopy(self.eval_policies[:len(self.nash_probs)]), copy.deepcopy(self.nash_probs)

    # Perform a warm-up for NeuPL.
    def warmup(self):
        print("NeuPL warming up!")
        self.runners[0].save()
        