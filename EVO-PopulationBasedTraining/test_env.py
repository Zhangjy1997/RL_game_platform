# import argparse
# import time
# import os
# import numpy as np

# from stable_baselines3 import PPO
# import pybullet_envs
# import gym
# from stable_baselines3.common.evaluation import evaluate_policy
# from tensorboardX import SummaryWriter


# def main():
#     env = gym.make('AntBulletEnv-v0')
#     # env = gym.make('Walker2DBulletEnv-v0')
#     # env = gym.make('HopperBulletEnv-v0')
#     env.reset()

#     counter = 0
#     while True:
#         counter += 1
#         action = env.action_space.sample()
#         nxt_obs, reward, done, info = env.step(action)
#         print(done, counter, info)
#         if done:
#             exit(0)

# if __name__ == '__main__':
#     main()



import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()