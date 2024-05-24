#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket
from onpolicy.utils.plot3d_test import plot_track, plot_git, get_track_gif
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as Multi_mix_policy
import json

# third-party packages
import numpy as np
import setproctitle
import torch
import wandb
import copy

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.uav.UAV_env import UAVEnv
from onpolicy.algorithms.NeuPL.train_NeuPL_sigma import find_unique_rows
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as empty_Policy
from onpolicy.algorithms.NeuPL.Population_eval.payoff_eval import get_policy_from_dir
# from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_E2P_3Doptimal as Evader_rule_policy
# from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_P2E_straight as Pursuer_rule_policy
from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_P2E_3Doptimal as Pursuer_rule_policy
from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_E2P_3Doptimal as Evader_rule_policy
from onpolicy.algorithms.NeuPL.eval_match import eval_match_uav as evaluator

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

def restore_eval_policy(policy, model_dir, label_str, use_mixer = True):
    """Restore policy's networks from a saved model."""
    policy_actor_state_dict = torch.load(str(model_dir) + '/actor_' + label_str + '.pt')
    policy.actor.load_state_dict(policy_actor_state_dict)
    if use_mixer:
        policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer_' + label_str + '.pt')
        policy.mixer.load_state_dict(policy_mixer_state_dict)

def creat_empty_policy(args, envs, device):
    empty_policy = empty_Policy(args,
                        envs.world.obs_space[0],
                        envs.world.obs_space[0],
                        envs.world.act_space[0],
                        device)
    return empty_policy

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
    parser.add_argument("--file_path", type=str, default="")
    
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

    parser.add_argument("--team_name", type=str, default="evader")
    parser.add_argument("--oppo_name", type=str, default="pursuer")
    parser.add_argument("--sigma_layer_N",type=int, default=1)

    # NeuPL setting
    parser.add_argument("--population_size", type=int, default=5)
    parser.add_argument("--runner_num", type=int, default=1)
    parser.add_argument("--global_steps", type=int, default=0)
    parser.add_argument("--eval_episode_num", type=int, default=10)
    parser.add_argument("--total_round", type=int, default=10)
    parser.add_argument("--channel_interval", type=int, default=10)
    parser.add_argument("--use_mix_policy", action='store_true', default=False)
    parser.add_argument("--use_inherit_policy", action='store_false', default=True)
    parser.add_argument("--use_warmup", action='store_false', default=True)
    parser.add_argument("--single_round", action='store_true', default=False)
    parser.add_argument("--policy_backup_dir", type=str, default=None)
    parser.add_argument("--begin_inx", type=int, default=1)
    parser.add_argument("--role_number", type=int, default=0)
    parser.add_argument("--use_share_policy", action='store_false', default=True)

    parser.add_argument("--use_track_recorder", action='store_true', default=False)
    parser.add_argument("--use_payoff_eval", action='store_true', default=False)
    parser.add_argument("--track_n", type=int, default=1)
    parser.add_argument("--population_type", type=str, default="neupl")
    parser.add_argument("--use_random_policy", action='store_true', default=False)
                        
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
        device = torch.device("cuda:0")
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
                         group=all_args.scenario_name,
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
    
    print("run_dir=",run_dir)

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

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.uav_dual_runner import UAV_Dual_Runner as Runner
    else:
        from onpolicy.runner.separated.uav_dual_runner import UAV_Dual_Runner as Runner

    # env init
    all_args.team_name = 'pursuer'
    all_args.oppo_name = 'evader'
    envs_p = make_train_env(all_args)
    eval_envs_p = make_eval_env(all_args) if all_args.use_eval else None

    eval_match_envs = make_train_env(all_args)

    mat_path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataV3.mat'
    grid = (np.linspace(25, 205, 46), np.linspace(-100, 100, 51), np.linspace(0, 100, 26),
        np.linspace(-5, 205, 36), np.linspace(-5, 205, 36))
    
    if all_args.use_random_policy:
        print("Use a random policy as the initial policy!")

    policies_p1 = []
    share_policies = []
    policy_anchor = []

    with open(str(all_args.model_dir)+'/policy_config.json', 'r') as f:
        params = json.load(f)

    for key, value in params.items():
        setattr(all_args, key, value)

    for i in range(all_args.population_size):
        policy_p1 = empty_Policy(all_args,
                            envs_p.observation_space[0],
                            envs_p.observation_space[0],
                            envs_p.action_space[0],
                            device)
        if i == 0:
            policy_rule_p = Pursuer_rule_policy(policy_p1, max_vel=7, mat_path = mat_path, grid = grid , device = device)
            if all_args.use_random_policy:
                policy_anchor.append(policy_p1)
                policies_p1.append(policy_p1)
            else:
                policy_anchor.append(policy_rule_p)
                policies_p1.append(policy_rule_p)
            
            share_policies.append(policy_p1)
        else:
            policies_p1.append(policy_p1)

    policies_p2 = []

    all_args.team_name = 'evader'
    all_args.oppo_name = 'pursuer'
    envs_e = make_train_env(all_args)
    eval_envs_e = make_train_env(all_args) if all_args.use_eval else None

    for i in range(all_args.population_size):
        policy_p2 = empty_Policy(all_args,
                    envs_e.observation_space[0],
                    envs_e.observation_space[0],
                    envs_e.action_space[0],
                    device)
        
        if i == 0:
            # mat_path = "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattackerV2.mat"
            # grid = (np.linspace(10, 210, 101), np.linspace(-100, 100, 101), np.linspace(-100, 100, 101))
            # mat_path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattackerV3.mat'
            # grid = (np.linspace(25, 205, 46), np.linspace(-100, 100, 51), np.linspace(0, 100, 26),
            #         np.linspace(-5, 205, 36), np.linspace(-5, 205, 36))
            policy_rule_e = Evader_rule_policy(policy_p2, max_vel=7, mat_path=mat_path, grid=grid, device = device)
            # policy_rule_e = Evader_rule_policy(policy_p2, max_vel=7, device=device)
            if all_args.use_random_policy:
                policy_anchor.append(policy_p2)
                policies_p2.append(policy_p2)
            else:
                policy_anchor.append(policy_rule_e)
                policies_p2.append(policy_rule_e)

            share_policies.append(policy_p2)
        else:
            policies_p2.append(policy_p2)


    # num_agents = envs_e.world.num_team
    # all_args.num_agents = num_agents
    # all_args.use_mixer = True if num_agents >= 2 else False
    
    role_name = ["pursuer", "evader"]
    role_dict = dict()
    role_dict["player1"] = "pursuer"

    round_str = all_args.model_dir

    policies_pursuer = get_policy_from_dir(all_args, eval_match_envs, round_str, policy_anchor[0], role_dict, device = device)
    role_dict = dict()
    role_dict["player2"] = "evader"
    policies_evader = get_policy_from_dir(all_args, eval_match_envs, round_str, policy_anchor[1], role_dict, device = device)
    print("number of policies = ", [len(policies_pursuer), len(policies_evader)])

    # for i in range(6):
    #     oppo_policies.append(creat_oppo_policy(all_args, eval_envs, pre_load_dir, str(eval_envs.world.oppo_name) + str(i+1), device))

    # oppo_probs = np.array([1, 1, 1, 1, 1, 1], dtype=float)

    # eval_envs.world.oppo_policy = Multi_mix_policy(all_args.n_rollout_threads, oppo_policies, probs= oppo_probs)
    
    #eval_envs.world.oppo_policy = creat_oppo_policy(all_args, eval_envs, pre_load_dir, str(eval_envs.world.oppo_name) + str(all_args.evader_num), device)

    # num_agents = eval_envs.world.num_team
    # all_args.num_agents = num_agents
    
    all_args.episode_length = envs_p.episode_length

    eval = evaluator(policies_pursuer, policies_evader, eval_match_envs)

    # all_args.use_mixer = True if num_agents >= 2 else False

    if all_args.use_track_recorder:
        for kk in range(all_args.track_n):
            print("round {}/{}".format(kk, all_args.track_n))
            obs_array, info_array = eval.get_track_array()
            for i in range(len(policies_pursuer)):
                for j in range(len(policies_evader)):
                    track_length = len(obs_array[i][j])
                    pursuer_track = np.zeros((track_length, eval_match_envs.world.num_team,3))
                    evader_track = np.zeros((track_length,eval_match_envs.world.num_oppo, 3))
                    for k in range(track_length):
                        for p in range(pursuer_track.shape[1]):
                            pursuer_track[k][p] = obs_array[i][j][k][0][p][0:3]
                        evader_track[k][0] = obs_array[i][j][k][0][0][3*eval_match_envs.world.num_team:]
                    folder_path = os.path.join(all_args.file_path, "pursuer_" + str(i) + " vs evader_" + str(j))
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                        print("Folder created:", folder_path)
                    else:
                        print("Folder already exists:", folder_path)

                    print("save gif!")
                    get_track_gif(pursuer_track, evader_track, folder_path, info_array[i][j], "pursuer_" + str(i) + " vs evader_" + str(j))

    # else:
    #     runner.calu_win_prob(20)
    #     print("win_num: ", runner.total_pursuer_win , runner.total_evader_win)
    #     print("win_prob: ", runner.total_pursuer_win/runner.total_round)
    #     print("total_round: ", runner.total_round)
    #     print("pursuer policy: {}, evader policy: {}".format(all_args.pursuer_num,all_args.evader_num))

    # post process
    eval_match_envs.close()
    if all_args.use_eval and eval_match_envs is not envs_p:
        eval_match_envs.close()

    print("use_wandb=",all_args.use_wandb)

    if all_args.use_wandb:
        run.finish()
    # else:
    #     runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    #     runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
