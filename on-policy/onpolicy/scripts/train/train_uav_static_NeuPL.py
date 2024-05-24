#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket
import copy

# third-party packages
import numpy as np
import setproctitle
import torch
import wandb

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.uav.UAV_env import UAVEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as Policy_module
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as mixing_policy
from onpolicy.algorithms.NeuPL.train_NeuPL_WF import find_unique_rows, compact_dictionaries, restore_eval_policy
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as Multi_mix_policy
from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_P2E_straight as Pursuer_rule_policy
from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_E2L_straight as Evader_rule_policy

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "UAV":
                ## TODO: pass hyper-parameters to the environment
                env = UAVEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env()
    return get_env_fn(0)


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "UAV":
                ## TODO: pass hyper-parameters to the environment
                env = UAVEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env()
    return get_env_fn(0)


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str,
                        default="simple_uav", 
                        help="which scenario to run on.")
    parser.add_argument("--num_agents", type=int, default=3,
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
                        default=16, help="hidden size of encoder")
    parser.add_argument("--proprio_shape", type=int, default=13,
                        help="proprio_shape")
    parser.add_argument("--teammate_shape", type=int, default=7,
                        help="teammate")
    parser.add_argument("--opponent_shape", type=int, default=3,
                        help="opponent_shape")
    parser.add_argument("--n_head", type=int, default=4, help="n_head")
    parser.add_argument("--d_k", type=int, default= 16, help="d_k")
    parser.add_argument("--attn_size", type=int, default=16, help="attn_size")
    parser.add_argument("--sigma_layer_N",type=int, default=1)
    parser.add_argument("--sigma_encoder_layer_N",type=int, default=1)

    parser.add_argument("--team_name", type=str, default="evader")
    parser.add_argument("--oppo_name", type=str, default="pursuer")

    parser.add_argument("--runner_num", type=int, default=1)
    parser.add_argument("--channel_interval", type=int, default=1)
    parser.add_argument("--global_steps", type=int, default=0)
    parser.add_argument("--eval_episode_num", type=int, default=10)
    parser.add_argument("--total_round", type=int, default=10)
    parser.add_argument("--use_mix_policy", action='store_true', default=False)
    parser.add_argument("--oppotent_name", type = str, default="")
    parser.add_argument("--population_size", type = int, default=1)
    parser.add_argument("--use_share_policy", action='store_false', default=True)
    parser.add_argument("--population_type", type=str, default="neupl")
    parser.add_argument("--policy_backup_dir", type=str, default="")
    parser.add_argument("--policy_restore_dir", type=str, default=None)
    
                        
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        # device = torch.device("cuda:0")
        device = torch.device("cuda")
        # torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        # torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                        #  entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name="-".join([
                            all_args.algorithm_name,
                            all_args.experiment_name,
                            "seed" + str(all_args.seed)
                         ]),
                         group="NeuPL_simple",
                        #  dir=str(run_dir),
                         job_type="training",
                         reinit=False)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle("-".join([
        all_args.env_name, 
        all_args.scenario_name, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]) + "@" + all_args.user_name)
    
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    if all_args.share_policy:
        from onpolicy.runner.shared.uav_sigma_runner import UAV_Sigma_Runner as Runner
    else:
        from onpolicy.runner.separated.uav_sigma_runner import UAV_Sigma_Runner as Runner

    if all_args.population_type == "neupl":
        discrete_policy = False
    elif all_args.population_type == "frozen_policy":
        discrete_policy = True
    else:
        raise NotImplementedError

    probs_p1 = np.load(os.path.join(all_args.policy_backup_dir, 'probs_p1.npy'))
    probs_p2 = np.load(os.path.join(all_args.policy_backup_dir, 'probs_p2.npy'))
    effect_p1_probs, effect_p1_map = find_unique_rows(matrix=probs_p1, max_num=all_args.population_size)
    effect_p2_probs, effect_p2_map = find_unique_rows(matrix=probs_p2, max_num=all_args.population_size)

    effect_p1_ids, effect_p2_ids = effect_p2_probs, effect_p1_probs
    effect_p2_id_map, effect_p1_id_map = compact_dictionaries(effect_p1_map, effect_p2_map)

    effect_p2_id_inx = np.arange(1, len(effect_p2_ids) + 1 ,1)
    effect_p1_id_inx = np.arange(1, len(effect_p1_ids) + 1 ,1)

    policy_num_p1 = effect_p1_ids.shape[0] + 1
    policy_num_p2 = effect_p2_ids.shape[0] + 1 

    # env init
    all_args.team_name = 'pursuer'
    all_args.oppo_name = 'evader'
    envs_p = make_train_env(all_args)
    eval_envs_p = make_eval_env(all_args)

    pursuer_policy_list = []
    for i in range(policy_num_p1):
        empty_policy = Policy_module(all_args,
                            envs_p.world.observation_space[0],
                            envs_p.world.observation_space[0],
                            envs_p.world.action_space[0],
                            device)
        if i==0:
            pos_e = eval_envs_p.world.num_team * 3
            pursuer_policy_list.append(Pursuer_rule_policy(empty_policy, max_vel=5, pos_e_inx=pos_e, device=device))
        else:
            if discrete_policy:
                restore_eval_policy(empty_policy, all_args.policy_backup_dir, 'pursuer', use_mixer = empty_policy.use_mixer, head_str=("frozen_policy_" + str(i)))
            else:
                restore_eval_policy(empty_policy, all_args.policy_backup_dir, 'pursuer', use_mixer = empty_policy.use_mixer)
            p2_sigma = torch.from_numpy(effect_p1_ids[i-1])
            # set sigma of player2's j-th policy
            empty_policy.set_sigma(np.tile(p2_sigma,(1,1)))
            empty_policy.set_fusion_false()
            pursuer_policy_list.append(empty_policy)

    # all_args.proprio_shape, all_args.teammate_shape, all_args.opponent_shape =  envs_p.world.sub_role_shape[all_args.team_name]['proprio_shape'], \
    #                                                                             envs_p.world.sub_role_shape[all_args.team_name]['teammate_shape'], \
    #                                                                             envs_p.world.sub_role_shape[all_args.team_name]['opponent_shape']
    
    # print("{} : p_shape {}, t_shape {}, o_shape {}.".format(all_args.team_name, all_args.proprio_shape, all_args.teammate_shape, all_args.opponent_shape))
    # num_agents = envs_p.world.num_team
    # all_args.num_agents = num_agents
    # all_args.use_mixer = True if num_agents >= 2 else False
    num_agents = envs_p.world.num_team
    all_args.episode_length = envs_p.episode_length

    if all_args.oppotent_name == "evader":
        config = {
            "all_args": all_args,
            "envs": envs_p,
            "eval_envs": eval_envs_p,
            "num_agents": envs_p.world.num_team,
            "device": device,
            "run_dir": run_dir
        }

        runner_p = Runner(config)

        if all_args.policy_restore_dir is not None:
            runner_p.inherit_policy(all_args.policy_restore_dir, "pursuer")

    policies = []

    all_args.team_name = 'evader'
    all_args.oppo_name = 'pursuer'
    envs_e = make_train_env(all_args)
    eval_envs_e = make_train_env(all_args)

    # all_args.proprio_shape, all_args.teammate_shape, all_args.opponent_shape =  envs_p.world.sub_role_shape[all_args.team_name]['proprio_shape'], \
    #                                                                             envs_p.world.sub_role_shape[all_args.team_name]['teammate_shape'], \
    #                                                                             envs_p.world.sub_role_shape[all_args.team_name]['opponent_shape']
    
    # print("{} : p_shape {}, t_shape {}, o_shape {}.".format(all_args.team_name, all_args.proprio_shape, all_args.teammate_shape, all_args.opponent_shape))

    #for i in range(6):
    #    policies.append(creat_oppo_policy(all_args, envs_e, pre_load_dir, "pursuer" + str(i+1), device))
    #print(policies[0].obs_space)
    
    # policies.append(oppo_policy_rule)

    num_agents = envs_e.world.num_team
    # all_args.num_agents = num_agents
    # all_args.use_mixer = True if num_agents >= 2 else False

    # mat_path = "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattackerV2.mat"
    # grid = (np.linspace(10, 210, 101), np.linspace(-100, 100, 101), np.linspace(-100, 100, 101))
    # mat_path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattackerV3.mat'
    # grid = (np.linspace(25, 205, 46), np.linspace(-100, 100, 51), np.linspace(0, 100, 26),
    #         np.linspace(-5, 205, 36), np.linspace(-5, 205, 36))

    # pos_p_in = all_args.proprio_shape + (envs_e.world.num_team - 1) *all_args.teammate_shape
    # runner_p.envs.world.oppo_policy = Evader_rule_policy(oppo_policy_empty, max_vel=7, mat_path=mat_path, grid=grid, device = device)
    
    evader_policy_list = []
    for i in range(policy_num_p2):
        empty_policy = Policy_module(all_args,
                            envs_p.world.oppo_obs_space[0],
                            envs_p.world.oppo_obs_space[0],
                            envs_p.world.oppo_act_space[0],
                            device)
        if i==0:
            evader_policy_list.append(Evader_rule_policy(copy.deepcopy(empty_policy), max_vel=7, device=device))
        else:
            if discrete_policy:
                restore_eval_policy(empty_policy, all_args.policy_backup_dir, 'evader', use_mixer = empty_policy.use_mixer, head_str=("frozen_policy_" + str(i)))
            else:
                restore_eval_policy(empty_policy, all_args.policy_backup_dir, 'evader', use_mixer = empty_policy.use_mixer)
            p1_sigma = torch.from_numpy(effect_p2_ids[i-1])
            # set sigma of player2's j-th policy
            empty_policy.set_sigma(np.tile(p1_sigma,(1,1)))
            empty_policy.set_fusion_false()
            evader_policy_list.append(empty_policy)
    
    if all_args.oppotent_name == "pursuer":
        config = {
            "all_args": all_args,
            "envs": envs_e,
            "eval_envs": eval_envs_e,
            "num_agents": envs_e.world.num_team,
            "device": device,
            "run_dir": run_dir
        }

        runner_e = Runner(config)
        if all_args.policy_restore_dir is not None:
            runner_e.inherit_policy(all_args.policy_restore_dir, "evader")

    mix_policy_p1 = mixing_policy(all_args.n_rollout_threads, pursuer_policy_list)
    mix_policy_p2 = mixing_policy(all_args.n_rollout_threads, evader_policy_list)

    # runner_e.envs.world.oppo_policy = Pursuer_rule_policy(oppo_policy_empty, max_vel = 7, pos_e_inx = 3* envs_e.world.num_oppo, device=device)

    print("Episode length is: ", all_args.episode_length)
    print("run_dir = ", str(wandb.run.dir))

    #trainer_frame.run()
    if all_args.oppotent_name == "pursuer":
        runner_e.envs.world.oppo_policy = mix_policy_p1
        runner_e.set_id_sigma(effect_p2_ids, effect_p2_id_inx)
        runner_e.run()
    elif  all_args.oppotent_name == "evader":
        runner_p.envs.world.oppo_policy = mix_policy_p2
        runner_p.set_id_sigma(effect_p1_ids, effect_p1_id_inx)
        runner_p.run()
    else:
        raise NotImplementedError
    
    # post process
    envs_p.close()
    envs_e.close()
    if eval_envs_p is not envs_p:
        eval_envs_p.close()
    if eval_envs_e is not envs_e:
        eval_envs_e.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner_e.writter.export_scalars_to_json(str(runner_e.log_dir + '/summary.json'))
        runner_e.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
