import os
import time
import numpy as np
import random

from flightgym import QuadrotorEnv_v0
from flightgym import MultiQuadrotorEnv_v0
from flightgym import QuadrotorPIDVelCtlEnv_v0
from flightgym import QuadrotorMPCVelCtlEnv_v0
from flightgym import MultiQuadrotorPIDVelCtlEnv_v0
from flightgym import MultiQuadrotorMPCVelCtlEnv_v0
from flightgym import MapStringFloatVector
from flightgym import MapStringFloat

def main():
  quad_env = QuadrotorEnv_v0()
  # quad_env = MultiQuadrotorEnv_v0()
  # quad_env = QuadrotorPIDVelCtlEnv_v0()
  # quad_env = QuadrotorMPCVelCtlEnv_v0()
  # quad_env = MultiQuadrotorPIDVelCtlEnv_v0()
  # quad_env = MultiQuadrotorMPCVelCtlEnv_v0()
  quad_env.setSeed(30)

  quad_env.connectUnity()

  num_envs = quad_env.getNumOfEnvs()
  obs_dim = quad_env.getObsDim()
  num_agent = quad_env.getNumAgent()
  act_dim = quad_env.getActDim()
  # print("Obs Dim = ", obs_dim)
  # print("Act Dim = ", act_dim)
  # print("Rew Dim = ", num_agent)

  obs = np.zeros(shape=(num_envs, obs_dim), dtype=np.float32)
  last_obs = np.zeros(shape=(num_envs, obs_dim), dtype=np.float32)
  quad_env.reset(obs)

  
  time.sleep(1)
  
  for i in range(10000):
    # time.sleep(0.1)
    # print(i)
    start = time.time()
    act = np.random.uniform(-1, 1, size=(num_envs, act_dim)) * 0.01
    act = act.astype(np.float32)
    # for j in range(num_envs):
    #   # act[j][0] += 9.8 # for body-rate and collective thrust control
    #   # act[j][1] = 0.0 # for velocity control
    #   act[j][2] = -0.5 # for velocity control
    #   act[j][6] = -0.5
    #   act[j][10] = -0.5
      # act[j][3] += -0.2 # for velocity control
    reward = np.zeros(shape=(num_envs, num_agent), dtype=np.float32)
    done = np.zeros(shape=(num_envs, num_agent), dtype=np.bool)
    # info = np.zeros(shape=(num_envs, len(quad_env.getExtraInfoNames())), dtype=np.float32)
    info = MapStringFloatVector()
    for i in range(num_envs):
      info.append(MapStringFloat())
    # print("Done = {}".format(done))
    reward = quad_env.step(act, obs, reward, done, info, last_obs)
    for inf in info:
      if 'TimeLimit.truncated' in inf.keys():
        print(list(inf.keys()))
        inf['terminal_observation'] = last_obs
      
    end = time.time()
    # print("Simulation Frequency is {}".format(1.0/(end-start)))

if __name__ == "__main__":
    main()
