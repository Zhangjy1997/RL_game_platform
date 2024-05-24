#!/usr/bin/env python3
from ruamel.yaml import YAML, dump, RoundTripDumper

#
import os
import math
import argparse
import numpy as np
import torch

#
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy as MlpPolicy
from stable_baselines3.common.vec_env import FlightEnvVec
import stable_baselines3.common.utils as U
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
#
from flightgym import QuadrotorPIDVelCtlEnv_v0, QuadrotorEnv_v0


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default='./saved/quadrotor_env.zip',
                        help='trained weight path')
    return parser


def main():
    args = parser().parse_args()
    cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    if args.render:
        cfg["env"]["render"] = "yes"
    else:
        cfg["env"]["render"] = "no"

    env = FlightEnvVec(QuadrotorEnv_v0(100, 100, False))

    tmp_path = "/home/qiyuan/workspace/stable_baselines_sac_log/"
    # set up logger
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

    if args.train:
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=1000,
            save_path=tmp_path,
            name_prefix="rl_model",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        # save the configuration and other files
        model = SAC(
            policy=MlpPolicy,  # check activation function
            policy_kwargs=dict(
                net_arch=[128, 128]),
            env=env,
            # clip_range=0.2,
            learning_rate=1e-4,
            gamma=0.99,
            gradient_steps=2,
            # ent_coef=0.01,
            learning_rate=1e-4,
            # vf_coef=0.5,
            # max_grad_norm=0.5,
            # batch_size=128,
            verbose=1,
        )

        # tensorboard
        # Make sure that your chrome browser is already on.
        # TensorboardLauncher(saver.data_dir + '/PPO2_1')

        # PPO run
        # Originally the total timestep is 5 x 10^8
        # 10 zeros for nupdates to be 4000
        # 1000000000 is 2000 iterations and so
        # 2000000000 is 4000 iterations.
        # logger.configure(folder=saver.data_dir)
        model.set_logger(new_logger)
        model.learn(total_timesteps=int(6000000), callback=checkpoint_callback)
        model.save(tmp_path)

    # # Testing mode with a trained weight
    else:
        model = SAC.load(tmp_path + 'rl_model_9400000_steps.zip')
        eval_env = FlightEnvVec(QuadrotorEnv_v0(1, 1, True))
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")


if __name__ == "__main__":
    main()
