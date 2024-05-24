#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket

# third-party packages
import numpy as np
import setproctitle
import torch
import wandb
import copy
import json
import re

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.HitBalls.HitBalls import PlankAndBall as Env
from onpolicy.envs.HitBalls.Plank_Ball import PlankAndBallENV
from onpolicy.algorithms.NeuPL.Policy_prob_matrix import Nash_matrix as graph_solver
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as Policy_module
from onpolicy.envs.HitBalls.Plank_ball_policy import Policy_P2B_straight
from onpolicy.algorithms.NeuPL.eval_match import eval_ball_cross_match as evaluator
from onpolicy.algorithms.NeuPL.Population_eval.payoff_eval import population_payoff

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "BALL":
                ## TODO: pass hyper-parameters to the environment
                env = PlankAndBallENV(Env(all_args.n_rollout_threads))
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
            if all_args.env_name == "BALL":
                ## TODO: pass hyper-parameters to the environment
                env = PlankAndBallENV(Env(all_args.n_rollout_threads))
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env()
    return get_env_fn(0)

def creat_empty_policy(args, envs, device):
    empty_policy = Policy_module(args,
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

    parser.add_argument("--use_payoff_eval", action='store_true', default=False)
    parser.add_argument("--track_n", type=int, default=1)
    parser.add_argument("--population_type", type=str, default="neupl")

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

    # env init
    envs = make_train_env(all_args)
    #eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    eval_envs = make_eval_env(all_args)
    all_args.episode_length = envs.episode_length

    empty_policy = Policy_module(all_args,
                            envs.world.observation_space[0],
                            envs.world.observation_space[0],
                            envs.world.action_space[0],
                            device)

    indices = range(9,11)
    policy_rule_p = Policy_P2B_straight(empty_policy, envs.world.max_vel_plank, envs.world.max_heading_plank, envs.world.sim_dt, indices, device=device)

    policy_rule_e = copy.deepcopy(policy_rule_p)

    eval = evaluator([policy_rule_p], [policy_rule_e], eval_envs)

    player1_dir = '/home/qiyuan/workspace/policy_backup/neupl_ball_20240324'
    player2_dir = '/home/qiyuan/workspace/policy_backup/neupl_PSRO_ball_20240324_2/round16'

    player1_dir_list = [os.path.join(player1_dir, "round" + str(i+1)) for i in range(16, 24)]
    player2_dir_list = [player2_dir for i in range(8)]

    payoff_ , prob_p1, prob_p2 = population_payoff(all_args, player1_dir_list, player2_dir_list, ["plank", "plank"], eval, [policy_rule_p, policy_rule_e], device=device)

    print("RPP = ", payoff_)
    np.save(os.path.join('/home/qiyuan/workspace/plot/20240328','RPP_20240327.npy'), payoff_)
    
        

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    print("use_wandb=",all_args.use_wandb)

    if all_args.use_wandb:
        run.finish()
    # else:
    #     runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
    #     runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
