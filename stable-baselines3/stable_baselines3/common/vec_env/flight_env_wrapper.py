import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv
from flightgym import MapStringFloatVector, MapStringFloat
import gymnasium as gym
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn


class FlightEnvVec(VecEnv):
    #
    def __init__(self, impl):
        self.wrapper = impl
        self.num_obs = self.wrapper.getObsDim()
        self.num_acts = self.wrapper.getActDim()
        self.num_agent = self.wrapper.getNumAgent()
        self.wrapper.connectUnity()
        self._observation_space = spaces.Box(
            np.ones(self.num_obs) * -np.Inf,
            np.ones(self.num_obs) * np.Inf, dtype=np.float32)
        self._action_space = spaces.Box(
            low=np.ones(self.num_acts) * -1.,
            high=np.ones(self.num_acts) * 1.,
            dtype=np.float32)
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._last_observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self._reward = np.zeros([self.num_envs, self.num_agent], dtype=np.float32)
        self._done = np.zeros(([self.num_envs, self.num_agent]), dtype=np.bool)
        self._extraInfo = MapStringFloatVector([MapStringFloat() for _ in range(self.num_envs)])
        self.rewards = [[] for _ in range(self.num_envs)]

        self.max_episode_steps = self.wrapper.getEpisodeLength()
        print("Per-environment Episode Length is:", self.max_episode_steps)

    def seed(self, seed=0):
        self.wrapper.setSeed(seed)

    def step(self, action):
        self.wrapper.step(action, self._observation,
                          self._reward, self._done, self._extraInfo, self._last_observation)
        info = [dict(self._extraInfo[i]) for i in range(self.num_envs)]
        ## TODO: check this expression
        for i in range(self.num_envs):
            self.rewards[i].append(self._reward[i, 0])
            if self._done[i, 0]:
                eprew = sum(self.rewards[i])
                eplen = len(self.rewards[i])
                epinfo = {"r": eprew, "l": eplen}
                info[i]['episode'] = epinfo
                info[i]['last_observation'] = self._last_observation # add by wangchao
                self.rewards[i].clear()

        return self._observation.copy(), self._reward[:, 0].copy(), \
            self._done[:, 0].copy(), info.copy()

    def stepUnity(self, action, send_id):
        receive_id = self.wrapper.stepUnity(action, self._observation,
                                            self._reward[:, 0], self._done[:, 0], self._extraInfo, send_id)

        return receive_id

    def sample_actions(self):
        actions = []
        for i in range(self.num_envs):
            action = self.action_space.sample().tolist()
            actions.append(action)
        return np.asarray(actions, dtype=np.float32)

    def reset(self):
        self.wrapper.reset(self._observation)
        return self._observation.copy()

    def reset_and_update_info(self):
        return self.reset(), self._update_epi_info()

    def _update_epi_info(self):
        info = [{} for _ in range(self.num_envs)]

        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()
        return info

    def render(self, mode='human'):
        raise RuntimeError('This method is not implemented')

    def close(self):
        self.wrapper.close()

    def connectUnity(self):
        self.wrapper.connectUnity()

    def disconnectUnity(self):
        self.wrapper.disconnectUnity()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def start_recording_video(self, file_name):
        raise RuntimeError('This method is not implemented')

    def stop_recording_video(self):
        raise RuntimeError('This method is not implemented')

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def step_async(self):
        raise RuntimeError('This method is not implemented')

    def step_wait(self):
        raise RuntimeError('This method is not implemented')

    def get_attr(self, attr_name, indices=None):
        """
        Return attribute from vectorized environment.
        :param attr_name: (str) The name of the attribute whose value to return
        :param indices: (list,int) Indices of envs to get attribute from
        :return: (list) List of values of 'attr_name' in all environments
        """
        raise RuntimeError('This method is not implemented')

    def set_attr(self, attr_name, value, indices=None):
        """
        Set attribute inside vectorized environments.
        :param attr_name: (str) The name of attribute to assign new value
        :param value: (obj) Value to assign to `attr_name`
        :param indices: (list,int) Indices of envs to assign value
        :return: (NoneType)
        """
        raise RuntimeError('This method is not implemented')

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.
        :param method_name: (str) The name of the environment method to invoke.
        :param indices: (list,int) Indices of envs whose method to call
        :param method_args: (tuple) Any positional arguments to provide in the call
        :param method_kwargs: (dict) Any keyword arguments to provide in the call
        :return: (list) List of items returned by the environment's method call
        """
        raise RuntimeError('This method is not implemented')
    
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        # target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [True for _ in range(self.num_envs)]
    
    # def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
    #     indices = self._get_indices(indices)
    #     return [self.envs[i] for i in indices]

# def main():
#   import os, time
#   from ruamel.yaml import YAML, dump, RoundTripDumper
#   from flightgym import QuadrotorPIDVelCtlEnv_v0, QuadrotorMPCVelCtl_v0, QuadrotorEnv_v0
#   env = QuadrotorPIDVelCtlEnv_v0()
#   env = FlightEnvVec(env)

#   obs = env.reset()

#   print(obs)

#   for i in range(10000):
#     act = np.zeros(shape=(env.num_envs, env.num_acts), dtype=np.float32)
#     next_obs, rew, done, info = env.step(act)
#     time.sleep(0.01)

# if __name__ == "__main__":
#   main()
