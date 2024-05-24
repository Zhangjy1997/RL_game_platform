# TODO:

import numpy as np
import torch
import wandb
from onpolicy.algorithms.NeuPL.eval_match import eval_match_ball as eval_match
from onpolicy.algorithms.NeuPL.Policy_prob_matrix import Nash_matrix as prob_matrix
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as mixing_policy
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as Policy
import random
import os
import time
import copy

def restore_eval_policy(policy, model_dir, use_mixer = True):
    """Restore policy's networks from a saved model."""
    policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
    policy.actor.load_state_dict(policy_actor_state_dict)
    if use_mixer:
        policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer.pt')
        policy.mixer.load_state_dict(policy_mixer_state_dict)

def creat_policy(args, envs, save_dir, device, use_mixer):
    policy = Policy(args,
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_act_space[0],
                        device)
    restore_eval_policy(policy, save_dir, use_mixer)
    return policy

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

class Neural_population_learning:
    #   Neupl is used for a two-player zero-sum game scenario, consisting of a pair of policy population {\pi(a|o,sigma_1), \pi(a|o,sigma_2)}
    #   sigma_{i} represents the ID of player{i}'s policy, and it also signifies the mixed probability of opponent's policy

    def __init__(self, args, anchor_policies, shared_policies, eval_policies, runners, evaluator, role_names, save_dir, device = torch.device("cpu")):
        # Store the input arguments and parameters
        self.args = args
        self.policies_anchor = anchor_policies
        self.policies_shared = shared_policies
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
        self.use_calc_exploit = self.args.use_calc_exploit
        self.num_env_steps = self.args.num_env_steps
        self.policy_num = self.args.population_size
        self.p1_space = [role_names[0] + str(i) for i in range(self.policy_num)]
        self.graph_generator = prob_matrix(self.p1_space, self.p1_space)
        self.eval_policies[0] = self.policies_anchor[0]
        self.effect_id_map = {0:0}
        self.fast_update = self.args.use_fast_update
        if self.fast_update:
            self.update_T = self.args.update_T

    # Calculate effective policies for the current game state.
    def calu_effective_policy(self):

        # Calculate effective policies for player 1 and player 2 based on probability matrices.
        effect_probs, effect_map = find_unique_rows(matrix=self.graph_generator.p1_prob_mat, max_num=self.policy_num)
        
        self.effect_ids = effect_probs

        self.effect_id_inx = np.arange(1, len(self.effect_ids) + 1 ,1)

        self.effect_id_map, _ = compact_dictionaries(effect_map, copy.deepcopy(effect_map))

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
            begin_inx += 1
            train_done = False
            # Continue training until it is effective
            sub_train_round = 0
            while train_done == False:
                self.step_run()
                coe_k = min(6, max(3, sub_train_round))
                train_done = self.check_training_effectiveness(coe_k=coe_k) or True
                sub_train_round += 1
                print("train_done = ",train_done)
            
        self.eval_round_step(begin_inx=begin_inx)
        self.cal_graph(round_num=self.total_round)

    # Run a single training round
    def run_single_round(self, begin_inx = 1):
        self.eval_round_step(begin_inx=begin_inx)
        self.cal_graph(round_num=begin_inx)
        self.step_run()
        train_done = self.check_training_effectiveness(coe_k=3)


    # Perform a training step for the specified available training roles.
    def step_run(self):

        if self.fast_update:
            eval_range = len(self.effect_id_inx) + 1
            for i in range(self.update_T):
                print("{}/{} gradient steps".format(i,self.update_T))
                for j in range(1, eval_range):
                    restore_eval_policy(self.eval_policies[j], self.save_dir, self.policies_shared[0].use_mixer)
                    p1_sigma = torch.from_numpy(self.effect_ids[j-1])
                    # set sigma of player1's j-th policy
                    self.eval_policies[j].set_sigma(np.tile(p1_sigma,(1,1)))
                    self.eval_policies[j].set_fusion_false()


                self.mix_policy = mixing_policy(self.n_threads, self.eval_policies, device=self.device)

                self.runners[0].all_args.global_steps = self.g_step
                self.runners[0].envs.world.oppo_policy = self.mix_policy
                self.runners[0].set_id_sigma(self.effect_ids)
                self.runners[0].run()
                self.g_step += self.num_env_steps

        else:
            self.runners[0].all_args.global_steps = self.g_step
            self.runners[0].envs.world.oppo_policy = self.mix_policy
            self.runners[0].set_id_sigma(self.effect_ids)
            self.runners[0].run()
            self.g_step += self.num_env_steps


    # Check the effectiveness of the training
    def check_training_effectiveness(self, coe_k = 2):
        
        train_effective = True

        eval_values, standard_vaule = self.runners[0].get_payoff_sigma(self.n_eval_eps)
        print("eval_player1 = ", eval_values)
        for inx in range(1, len(self.effect_id_inx) + 1):
            prob_vector = self.effect_ids[inx-1]
            #  calculate the payoff before the training
            last_eval_vaule = np.dot(self.payoff_mat[inx], prob_vector)

            print("policy {} last_eval_vaule ={}".format(inx,last_eval_vaule))
            delta_vaule = eval_values[inx] - last_eval_vaule
            print("threshold = ", coe_k*standard_vaule[inx])
            train_effective = train_effective and (delta_vaule > -coe_k*standard_vaule[inx])

        return train_effective

    # Perform evaluation steps before a training round
    def eval_round_step(self, begin_inx):
        # cover_inx = max(1, min(begin_inx, self.policy_num))
        if begin_inx <= 1:
            eval_range = 1
        else:
            eval_range = len(self.effect_id_inx) + 1
        
        # Restore evaluation policies of player1 and player2
        for j in range(1, eval_range):
            restore_eval_policy(self.eval_policies[j], self.save_dir, self.policies_shared[0].use_mixer)
            p1_sigma = torch.from_numpy(self.effect_ids[j-1])
            # set sigma of player1's j-th policy
            self.eval_policies[j].set_sigma(np.tile(p1_sigma,(1,1)))
            self.eval_policies[j].set_fusion_false()

        self.eval.update_policy(self.eval_policies[0:eval_range])

        # Calculate win prob. matrices and payoff matrix
        self.payoff_mat_ = self.eval.get_win_prob_mat(self.n_threads, self.n_eval_eps)

    # Calculate and update the game graph based on evaluation results
    def cal_graph(self, round_num):
        # cover_inx = max(1, min(begin_inx, self.policy_num -1)) + 1
        dict_mat_p1 = dict()
        dict_mat_p2 = dict()

        # Check matrix sync to ensure the same payoff matrix is used in all machines
        # payoff_mat = (self.eval.win_num_mat - self.eval.lose_num_mat)/(self.n_threads*self.n_eval_eps)
        payoff_mat_ = copy.deepcopy(self.payoff_mat_)
        payoff_mat = payoff_mat_ - payoff_mat_.T

        print("payoff_mat = ", payoff_mat)

        # payoff_mat = np.pad(payoff_mat, ((0, self.policy_num - payoff_mat.shape[0]), (0, self.policy_num - payoff_mat.shape[1])), 'constant', constant_values=0)
        self.graph_generator.update_prob_matrix(payoff_mat,self.effect_id_map,self.effect_id_map)
        if self.use_calc_exploit:
            probs, _ = self.graph_generator.caul_prob_from_payoff(payoff_mat)
            self.nash_probs = np.array(probs[0])
        # dict_mat_p1["win_prob_mat_" + str(round_num)] = self.eval.win_prob_mat
        dict_mat_p1["probs_p1_mat_" + str(round_num)] = self.graph_generator.p1_prob_mat
        dict_mat_p1["payoff_p1_mat_" + str(round_num)] = payoff_mat
        dict_mat_p2["probs_p2_mat_" + str(round_num)] = self.graph_generator.p2_prob_mat

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
        self.calu_effective_policy()
        np.save(os.path.join(self.save_dir, 'probs_p1.npy'), self.graph_generator.p1_prob_mat)
        np.save(os.path.join(self.save_dir, 'probs_p2.npy'), self.graph_generator.p2_prob_mat)

        print("prob_p1=")
        print(self.graph_generator.p1_prob_mat)
        print("prob_p2=")
        print(self.graph_generator.p2_prob_mat)

        # Create mixing policies for player 1 and player 2
        self.mix_policy = mixing_policy(self.n_threads, self.eval_policies, device=self.device)


        if self.use_wandb:
            for k, v in dict_mat_p1.items():
                wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p1_space)})
            for k, v in dict_mat_p2.items():
                wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p1_space)})
            wandb.log({"train_rounds": round_num})
            wandb.log({"effect_policy_number_p1": len(self.effect_id_inx)})

    # Load effective policy id from saved files
    def load_effect_sigma(self):
        probs_p1 = np.load(os.path.join(self.save_dir, 'probs_p1.npy'))
        probs_p2 = np.load(os.path.join(self.save_dir, 'probs_p2.npy'))
        self.graph_generator.p1_prob_mat = probs_p1
        self.graph_generator.p2_prob_mat = probs_p2

        self.calu_effective_policy()

    def restore(self):
        self.load_effect_sigma()

    def get_sub_nash_policy(self):
        return copy.deepcopy(self.eval_policies[:len(self.nash_probs)]), copy.deepcopy(self.nash_probs)

    # Perform a warm-up for NeuPL.
    def warmup(self):
        print("NeuPL warming up!")
        self.runners[0].save()
        

