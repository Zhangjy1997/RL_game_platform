import numpy as np
import json
import os
import re
import torch
import copy

from onpolicy.algorithms.NeuPL.train_NeuPL_FNR import find_unique_rows, compact_dictionaries
from onpolicy.algorithms.NeuPL.train_NeuPL_FNR import restore_eval_policy as restore_2player_policy
from onpolicy.algorithms.NeuPL.train_FNR_symmetry import restore_eval_policy as restore_symmetry_policy
from onpolicy.algorithms.NeuPL.Policy_prob_matrix import Nash_matrix as solver
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as mixing_policy
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as shared_policy
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as single_policy


def extract_match(pattern, filename):
    match = re.search(pattern, filename)
    if match:
        return match.group(0), int(match.group(1)), int(match.group(2))
    return None  

def get_policy_from_dir(args, envs, policy_dir, anchor_policy, role_dict, device = torch.device("cpu")):
    with open(str(policy_dir)+'/policy_config.json', 'r') as f:
        params = json.load(f)

    for key, value in params.items():
        setattr(args, key, value)

    discrete_policy_2level = False

    if args.population_type == "neupl":
        discrete_policy = False
    elif args.population_type == "frozen_policy":
        discrete_policy = True
    elif "MFR" in args.population_type:
        discrete_policy = True
        discrete_policy_2level = True
        sub_policy_num = params["sub_policy_num"]
    else:
        print("wrong population type!")
        raise NotImplementedError
    
    if discrete_policy and discrete_policy_2level:
        pattern = r'frozen_policy_(\d+)_(\d+)'

        matches = []
        for filename in os.listdir(policy_dir):
            result = extract_match(pattern, filename)
            if result:
                matches.append(result)
        matches.sort(key=lambda x: (x[1], x[2]))
        frozen_head_strs = []
        frozen_state = np.zeros((args.population_size, sub_policy_num), dtype=bool)
        num_of_level_files = np.zeros(args.population_size, dtype=int)
        num_of_level_files[0] = 1
        for match, level_i, sub_level_j in matches:
            frozen_head_strs.append(match)
            frozen_state[level_i, sub_level_j] = True
            num_of_level_files[level_i] = max(num_of_level_files[level_i], sub_level_j + 1)

        np.save(os.path.join(str(policy_dir), "files_num_of_level.npy"), num_of_level_files)
        max_policy_num = len(frozen_head_strs) + 1


    elif discrete_policy:
        pattern = r'frozen_policy_(\d+)'
        max_value = -1
        for filename in os.listdir(policy_dir):
            match = re.search(pattern, filename)
            if match:
                value = int(match.group(1))
                if value > max_value:
                    max_value = value
        
        if max_value != -1:
            max_policy_num = max_value + 1
        else:
            max_policy_num = 1
    else:
        max_policy_num = args.population_size

    role_list = ["player1", "player2", "symmetry"]

    role_dict_set = set(role_dict.keys())
    role_list_set = set(role_list)

    if role_dict_set.issubset(role_list_set) == False:
        print("wrong role name!")
        raise NotImplementedError

    if "player1" in role_dict:
        probs = np.load(os.path.join(policy_dir, 'probs_p2.npy'))
    else:
        probs = np.load(os.path.join(policy_dir, 'probs_p1.npy'))

    if discrete_policy_2level:
        for key, value in  role_dict.items():
            frozen_ids = np.load(os.path.join(policy_dir, "frozen_id_" + str(value) + ".npy"))

    policy_num = min(args.population_size, max_policy_num)
    effect_ids, effect_map = find_unique_rows(matrix=probs, max_num=policy_num)
    policy_num = effect_ids.shape[0] + 1

    player_policy_list = []
    if args.use_population:
        if discrete_policy_2level:
            for i in range(args.population_size):
                if i == 0:
                    player_policy_list.append(anchor_policy)
                else:
                    for j in range(sub_policy_num):
                        if frozen_state[i][j]:
                            if "symmetry" in role_dict:
                                empty_policy = shared_policy(args,
                                            envs.world.observation_space[0],
                                            envs.world.observation_space[0],
                                            envs.world.action_space[0],
                                            device)
                                restore_symmetry_policy(empty_policy, policy_dir, 
                                                        use_mixer = empty_policy.use_mixer, 
                                                        head_str=("frozen_policy_" + str(i)+'_'+str(j)))
                            elif "player1" in role_dict:
                                empty_policy = shared_policy(args,
                                            envs.world.observation_space[0],
                                            envs.world.observation_space[0],
                                            envs.world.action_space[0],
                                            device)
                                restore_2player_policy(empty_policy, policy_dir, 
                                                       label_str=role_dict['player1'], 
                                                       use_mixer = empty_policy.use_mixer, 
                                                       head_str=("frozen_policy_" + str(i)+'_'+str(j)))
                            else:
                                empty_policy = shared_policy(args,
                                            envs.world.oppo_obs_space[0],
                                            envs.world.oppo_obs_space[0],
                                            envs.world.oppo_act_space[0],
                                            device)
                                restore_2player_policy(empty_policy, policy_dir, 
                                                       label_str=role_dict['player2'], 
                                                       use_mixer = empty_policy.use_mixer, 
                                                       head_str=("frozen_policy_" + str(i)+'_'+str(j)))
                            p1_sigma = torch.from_numpy(frozen_ids[i][j])
                            empty_policy.set_sigma(np.tile(p1_sigma,(1,1)))
                            empty_policy.set_fusion_false()
                            player_policy_list.append(empty_policy)

        else:
            for i in range(policy_num):
                if i==0:
                    player_policy_list.append(anchor_policy)
                else:
                    if "symmetry" in role_dict:
                        empty_policy = shared_policy(args,
                                    envs.world.observation_space[0],
                                    envs.world.observation_space[0],
                                    envs.world.action_space[0],
                                    device)
                        if discrete_policy:
                            restore_symmetry_policy(empty_policy, policy_dir, use_mixer = empty_policy.use_mixer, head_str=("frozen_policy_" + str(i)))
                        else:
                            restore_symmetry_policy(empty_policy, policy_dir, use_mixer = empty_policy.use_mixer)
                    elif "player1" in role_dict:
                        empty_policy = shared_policy(args,
                                    envs.world.observation_space[0],
                                    envs.world.observation_space[0],
                                    envs.world.action_space[0],
                                    device)
                        if discrete_policy:
                            restore_2player_policy(empty_policy, policy_dir, label_str=role_dict['player1'], use_mixer = empty_policy.use_mixer, head_str=("frozen_policy_" + str(i)))
                        else:
                            restore_2player_policy(empty_policy, policy_dir, label_str=role_dict['player1'], use_mixer = empty_policy.use_mixer)
                    else:
                        empty_policy = shared_policy(args,
                                    envs.world.oppo_obs_space[0],
                                    envs.world.oppo_obs_space[0],
                                    envs.world.oppo_act_space[0],
                                    device)
                        if discrete_policy:
                            restore_2player_policy(empty_policy, policy_dir, label_str=role_dict['player2'], use_mixer = empty_policy.use_mixer, head_str=("frozen_policy_" + str(i)))
                        else:
                            restore_2player_policy(empty_policy, policy_dir, label_str=role_dict['player2'], use_mixer = empty_policy.use_mixer)
                    p1_sigma = torch.from_numpy(effect_ids[i-1])
                    empty_policy.set_sigma(np.tile(p1_sigma,(1,1)))
                    empty_policy.set_fusion_false()
                    player_policy_list.append(empty_policy)

    else:
        for i in range(policy_num):
            if i==0:
                player_policy_list.append(anchor_policy)
            else:
                if "symmetry" in role_dict:
                    empty_policy = single_policy(args,
                                envs.world.observation_space[0],
                                envs.world.observation_space[0],
                                envs.world.action_space[0],
                                device)
                    if discrete_policy:
                        restore_symmetry_policy(empty_policy, policy_dir, use_mixer = empty_policy.use_mixer, head_str=("frozen_policy_" + str(i)))
                    else:
                        restore_symmetry_policy(empty_policy, policy_dir, use_mixer = empty_policy.use_mixer)
                elif "player1" in role_dict:
                    empty_policy = single_policy(args,
                                envs.world.observation_space[0],
                                envs.world.observation_space[0],
                                envs.world.action_space[0],
                                device)
                    if discrete_policy:
                        restore_2player_policy(empty_policy, policy_dir, label_str=role_dict['player1'], use_mixer = empty_policy.use_mixer, head_str=("frozen_policy_" + str(i)))
                    else:
                        restore_2player_policy(empty_policy, policy_dir, label_str=role_dict['player1'], use_mixer = empty_policy.use_mixer)
                else:
                    empty_policy = single_policy(args,
                                envs.world.oppo_obs_space[0],
                                envs.world.oppo_obs_space[0],
                                envs.world.oppo_act_space[0],
                                device)
                    if discrete_policy:
                        restore_2player_policy(empty_policy, policy_dir, label_str=role_dict['player2'], use_mixer = empty_policy.use_mixer, head_str=("frozen_policy_" + str(i)))
                    else:
                        restore_2player_policy(empty_policy, policy_dir, label_str=role_dict['player2'], use_mixer = empty_policy.use_mixer)
                player_policy_list.append(empty_policy)

    return player_policy_list

def population_payoff(args, player1_dir_list, player2_dir_list, role_names, evals, anchor_policies, device):
    if len(player1_dir_list) != len(player2_dir_list):
        print("The number of storage paths does not match!")
        raise NotImplementedError
    
    if role_names[0] == role_names[1]:
        symmetry_game = True
    else:
        symmetry_game = False
    
    payoff_array = np.zeros((len(player1_dir_list), 2))
    final_nash_probs_p1 = []
    final_nash_probs_p2 = []
    
    for i in range(len(player1_dir_list)):
        temp_p1_dir, temp_p2_dir = player1_dir_list[i], player2_dir_list[i]
        role_dict = dict()
        role_dict['symmetry' if symmetry_game else 'player1'] = role_names[0]
        policies_p1 = get_policy_from_dir(args, evals.envs, temp_p1_dir, anchor_policies[0], role_dict, device=device)

        role_dict = dict()
        role_dict['symmetry' if symmetry_game else 'player2'] = role_names[1]
        policies_p2 = get_policy_from_dir(args, evals.envs, temp_p2_dir, anchor_policies[1], role_dict, device=device)

        evals.update_policy(policies_p1, policies_p2)

        payoff_mat = evals.get_win_prob_mat(args.n_rollout_threads, args.eval_episode_num)

        nash_solver = solver(role_names, role_names)

        probs_, payoff_ = nash_solver.caul_prob_from_payoff(payoff_mat)

        payoff_array[i,0] = payoff_[0]
        payoff_array[i,1] = payoff_[1]
        final_nash_probs_p1.append(probs_[0])
        final_nash_probs_p2.append(probs_[1])

        print("{}/{} round complete".format(i+1, len(player1_dir_list)))
        print("payoff of player1 = ", payoff_[0])
        print("final probs player1 = ", probs_[0])
        print("final probs player2 = ", probs_[1])

    return payoff_array, final_nash_probs_p1, final_nash_probs_p2

def exploitability_policies(args, opponent_dir_list, runner ,role_names, anchor_policies, probs, device = torch.device("cpu")):
    if role_names[0] == role_names[1]:
        symmetry_game = True
    else:
        symmetry_game = False
    mixing_probs = copy.deepcopy(probs)
    role_dict = dict()
    role_dict['symmetry' if symmetry_game else 'player2'] = role_names[1]
    policies_oppo = get_policy_from_dir(args, runner.envs, opponent_dir_list, anchor_policies[0], role_dict, device=device)
    opponent_mix_policy = mixing_policy(args.n_rollout_threads, policies_oppo, probs= mixing_probs)
    runner.envs.world.oppo_policy = opponent_mix_policy
    runner.run()



