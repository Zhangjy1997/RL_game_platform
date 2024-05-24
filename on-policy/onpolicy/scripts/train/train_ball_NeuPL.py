#!/usr/bin/env python
# python standard libraries
import os
import re
from pathlib import Path
import sys
import socket
import fnmatch
import copy
import shutil
import glob

# third-party packages
import numpy as np
import setproctitle
import torch
import wandb
import json
import time

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.HitBalls.HitBalls import PlankAndBall as Env
from onpolicy.envs.HitBalls.Plank_Ball import PlankAndBallENV
from onpolicy.algorithms.NeuPL.eval_match import eval_match_ball_final as eval
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as empty_Policy
from onpolicy.algorithms.NeuPL.train_NeuPL_symmetry import Neural_population_learning
from onpolicy.envs.HitBalls.Plank_ball_policy import Policy_P2B_straight
# from onpolicy.algorithms.policy_DG.simple_policy_rule import Policy_E2P_3Doptimal as Evader_rule_policy
from onpolicy.algorithms.utils.ssh_client import send_files_scp, check_and_download_files

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


def find_max_round_folder(path):
    max_round = 0
    round_folder_pattern = re.compile(r'round(\d+)')

    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            match = round_folder_pattern.match(folder)
            if match:
                round_number = int(match.group(1))
                if round_number > max_round:
                    max_round = round_number

    if max_round == 0:
        os.makedirs(os.path.join(path, 'round1'))
        return 1
    else:
        return max_round

def check_files_exist(directory, match_str):
    has_str = False

    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, match_str):
            has_str = True

        if has_str:
            break

    return has_str

def copy_pt_files(source_dir, dest_dir, endstr):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for file in os.listdir(source_dir):
        if file.endswith(endstr):
            src_file = os.path.join(source_dir, file)
            dest_file = os.path.join(dest_dir, file)

            shutil.copy(src_file, dest_file)
            print(f"Copied: {src_file} to {dest_file}")

def delete_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"The folder '{folder_path}' has been deleted.")
    else:
        print(f"The folder '{folder_path}' does not exist or is not a directory.")

def delete_files_with_suffix(folder_path, end_str):
    pattern = os.path.join(folder_path, f'*{end_str}')

    files = glob.glob(pattern)

    for file in files:
        os.remove(file)
        print(f'Deleted: {file}')

def create_new_path(path1, path2):
    parts1 = [part for part in path1.split('/') if part]
    parts2 = [part for part in path2.split('/') if part]
    # print(parts1)
    # print(parts2)

    common_folder = None
    for part in parts1:
        if part in parts2[-1]:
            common_folder = part
            break

    if not common_folder:
        raise ValueError("No common folder found.")

    index = parts1.index(common_folder) + 1
    sub_path = '/'.join(parts1[index:])

    new_path = os.path.join(path2, sub_path)

    return new_path

def update_json(file_path, key, value):
    # Check if the file exists
    if os.path.exists(file_path):
        # If the file exists, read and update its content
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {file_path}")
            data = {}
    else:
        # If the file does not exist, create a new dictionary
        data = {}

    # Update or add the key-value pair
    data[key] = value

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

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

    parser.add_argument("--team_name", type=str, default="evader")
    parser.add_argument("--oppo_name", type=str, default="pursuer")
    parser.add_argument("--sigma_layer_N",type=int, default=1)
    parser.add_argument("--sigma_encoder_layer_N",type=int, default=1)

    # NeuPL setting
    parser.add_argument("--population_size", type=int, default=6)
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
    parser.add_argument("--use_calc_exploit", action='store_true', default=False)
    parser.add_argument("--use_fast_update", action='store_true', default=False)
    parser.add_argument("--update_T", type=int, default=100)
                        
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
                         #group=all_args.scenario_name,
                         group="NeuPL_ball",
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
        from onpolicy.runner.shared.ball_symmetry_runner import BALL_Sigma_Runner as Runner
    else:
        from onpolicy.runner.separated.ball_symmetry_runner import BALL_Sigma_Runner as Runner

    # env init
    envs_p = make_train_env(all_args)
    eval_envs_p = make_eval_env(all_args) if all_args.use_eval else None

    eval_match_envs = make_train_env(all_args)

    # all_args.num_agents = num_agents
    # all_args.use_mixer = True if num_agents >= 2 else False
    all_args.episode_length = envs_p.episode_length

    config = {
        "all_args": all_args,
        "envs": envs_p,
        "eval_envs": eval_envs_p,
        "num_agents": 1,
        "device": device,
        "run_dir": run_dir
    }
    
    runner_p = Runner(config)

    policies_p1 = []
    share_policies = []
    policy_anchor = []

    for i in range(all_args.population_size):
        policy_p1 = empty_Policy(all_args,
                            envs_p.observation_space[0],
                            envs_p.observation_space[0],
                            envs_p.action_space[0],
                            device)
        if i == 0:
            indices = range(9,11)
            policy_rule_p = Policy_P2B_straight(policy_p1, envs_p.world.max_vel_plank, envs_p.world.max_heading_plank, envs_p.world.sim_dt, indices, device=device)
            policy_anchor.append(policy_rule_p)
            share_policies.append(policy_p1)
            policies_p1.append(policy_rule_p)
        else:
            policies_p1.append(policy_p1)

    evaluator = eval(policies_p1, eval_match_envs)


    # num_agents = envs_e.world.num_team
    # all_args.num_agents = num_agents
    # all_args.use_mixer = True if num_agents >= 2 else False
    
    role_name = ["plank"]
    print("policy_backup_dir = ", all_args.policy_backup_dir)
    if all_args.use_wandb:
        NeuPL_trainer = Neural_population_learning(all_args, policy_anchor, share_policies,policies_p1, [runner_p], evaluator, role_name, str(wandb.run.dir), device=device)
        if all_args.policy_backup_dir is None:
            NeuPL_trainer.run()
        else:
            if not os.path.exists(all_args.policy_backup_dir):
                os.makedirs(all_args.policy_backup_dir)
                print(f"Folder '{all_args.policy_backup_dir}' created.")
            else:
                print(f"Folder '{all_args.policy_backup_dir}' already exists.")
            max_round = find_max_round_folder(all_args.policy_backup_dir)
            seek_role = True
            for i in range(max_round,0, -1):
                real_max_round = i
                round_dir = os.path.join(all_args.policy_backup_dir, "round"+str(i))
                if seek_role and check_files_exist(round_dir,'*.pt'):
                    copy_pt_files(round_dir, str(wandb.run.dir),'.pt')
                    seek_role = False
                if seek_role is False:
                    if check_files_exist(round_dir,'*p1.npy') and check_files_exist(round_dir,'*p2.npy'):
                        copy_pt_files(round_dir, str(wandb.run.dir),'p1.npy')
                        copy_pt_files(round_dir, str(wandb.run.dir),'p2.npy')
                        NeuPL_trainer.restore()
                    break
                # delete_folder(round_dir)
            if seek_role is True:
                real_max_round = 0
            print("real_max_round = ",real_max_round)
            if seek_role is False:
                NeuPL_trainer.runners[0].inherit_policy(str(wandb.run.dir))

            if real_max_round == 0:
                    NeuPL_trainer.warmup()
            
            for k in range(all_args.total_round):
                next_round_dir = os.path.join(all_args.policy_backup_dir, "round" + str(real_max_round+1))
                try:
                    os.mkdir(next_round_dir)
                    print(f"Folder '{next_round_dir}' created successfully.")
                except FileExistsError:
                    print(f"Folder '{next_round_dir}' already exists.")
                except OSError as error:
                    print(f"Creation of the folder '{next_round_dir}' failed due to: {error}")
                NeuPL_trainer.run_single_round(begin_inx=real_max_round+1)
                print("Next round!")
                real_max_round += 1
                round_dir = os.path.join(all_args.policy_backup_dir, "round" + str(real_max_round))
                copy_pt_files(str(wandb.run.dir), round_dir,'p1.npy')
                copy_pt_files(str(wandb.run.dir), round_dir,'p2.npy')
                copy_pt_files(str(wandb.run.dir), round_dir,'.pt')
    

        if all_args.single_round:
            NeuPL_trainer.run_single_round()

    #trainer_frame = train_alternate(all_args, all_args.total_round, runner_p, runner_e, str(wandb.run.dir), device)

    
    # post process
    envs_p.close()
    if all_args.use_eval and eval_envs_p is not envs_p:
        eval_envs_p.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner_p.writter.export_scalars_to_json(str(runner_p.log_dir + '/summary.json'))
        runner_p.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
