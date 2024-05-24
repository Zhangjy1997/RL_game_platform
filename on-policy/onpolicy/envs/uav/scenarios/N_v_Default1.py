import numpy as np
from onpolicy.envs.uav.scenarios.vec_env_wrapper import SimpleBase
from onpolicy.envs.uav.scenario import BaseScenarioUAV
from gym import spaces
import yaml
import copy
import time


def dict2vector(obs):
    observations = []
    for player in obs.keys():
        obs_player_wise = obs[player]
        observations_player_wise = []
        # parse proprio obs
        observations_player_wise += [obs_player_wise['proprioceptive'][0]['obs']]
        # parse teammate obs
        for teammate in obs_player_wise['exterprioceptive']['teammate']:
            observations_player_wise += [teammate['obs']]
        # parse opponent obs
        for opponent in obs_player_wise['exterprioceptive']['opponent']:
            observations_player_wise += [opponent['obs']]
        # parse context obs
        for context in obs_player_wise['exterprioceptive']['context']:
            observations_player_wise += [context['obs']]
        observations_player_wise = np.expand_dims(np.concatenate(observations_player_wise, axis=1), axis=1)
        observations.append(observations_player_wise)
    observations = np.concatenate(observations, axis=1)
    return observations

class NvDefault1(SimpleBase):
    def __init__(self, num_envs, num_threads, render):
        super().__init__(num_envs, num_threads, render)
        ## for debugging only
        self.debug = False
        if self.debug:
            self.obs_log = []
            self.rollout_counter = 0
            self.step_counter = 0

        observation_space = copy.deepcopy(self.observation_space)
        action_space = copy.deepcopy(self.action_space)
        evader_action_space = copy.deepcopy(self.action_space)
        for k in self.role_keys:
            if 'evader' in k:
                observation_space.pop(k)
                action_space.pop(k)
            if 'pursuer' in k:
                evader_action_space.pop(k)
        self.observation_space = list(observation_space.values())
        self.action_space = list(action_space.values())
        self.evader_action_space = list(evader_action_space.values())
        self.obs = None

    def _reset_action(self, act):
        assert act.shape[1] == self.num_pursuer_ * self.action_space[0].shape[0], 'wrong action dim!'
        # TODO: currently we randomly sample evader actions
        # evader_action = np.concatenate([np.expand_dims(np.concatenate([self.evader_action_space[i].sample() for i in range(self.num_evader_)]), axis=0) for _ in range(self.num_env_)], axis=0)
        evader_action = self.get_evader_action(const_speed=7, const_heading_rate=0.57/2)
        ## TODO: currently we clip the network action, in future ......
        #act = np.clip(act, -1, 1)
        action = np.concatenate([act, evader_action], axis=1)
        return action

    def get_observation_new(self):
        obs = super().get_observation()
        self.obs = copy.deepcopy(obs)
        if self.debug: tmp_obs = []
        for k in self.role_keys:
            if self.debug: tmp_obs.append(obs[k]['proprioceptive'][0]['obs'][0][:3])
            if 'evader' in k:
                obs.pop(k)
        if self.debug:
            self.obs_log.append(np.concatenate(tmp_obs))
        return dict2vector(obs)

    def get_reward_new(self):
        reward = super().get_reward()
        for k in self.role_keys:
            if 'evader' in k:
                reward.pop(k)
        return np.expand_dims(np.concatenate(list(reward.values()), axis=1), axis=-1)
    
    def get_done_new(self):
        done = super().get_done()
        for k in self.role_keys:
            if 'evader' in k:
                done.pop(k)
        return np.concatenate(list(done.values()), axis=1)
    
    def get_info_new(self):
        info = super().get_info()
        return info

    def step(self, act):
        action = self._reset_action(act)
        super().step(action)
        obs = self.get_observation_new()
        reward = self.get_reward_new()
        done = self.get_done_new()
        info = self.get_info_new()
        if self.debug: self.step_counter += 1
        if done[0].all() and self.debug:
            np.save('./' + str(self.rollout_counter) + '_observations.npy', np.array(self.obs_log))
            self.obs_log = []
        return obs, reward, done, info
    
    def reset(self):
        super().reset()
        if self.debug:
            self.step_counter = 0
            self.obs_log = []
        return self.get_observation_new()
    
    def get_evader_action(self, const_speed=None, const_heading_rate=None, random=False):
        if random:
            evader_action = np.concatenate([np.expand_dims(np.concatenate([self.evader_action_space[i].sample() for i in range(self.num_evader_)]), axis=0) for _ in range(self.num_env_)], axis=0)
            return evader_action

        ########## compute evader actions
        # compute evader position and velocity
        for k in self.role_keys:
            if 'evader' in k:
                obs = self.obs[k]['proprioceptive'][0]['obs'].transpose()
                position = obs[self.map['position']['inx']].transpose()
                orientation = obs[self.map['orientation']['inx']].transpose()

                # compute linear velocity
                position_x, position_y = position[:, 0:1], position[:, 1:2]
                horizontal_distance = np.expand_dims(np.linalg.norm(position[:, :2], axis=1), axis=1)
                sin_theta = -position_y / horizontal_distance
                cos_theta = -position_x / horizontal_distance
                velocity_x = const_speed * cos_theta
                velocity_y = const_speed * sin_theta
                velocity_z = 0 * cos_theta

                # compute angular velocity [z-axis]
                orientation_z = orientation[:, 2:3]
                orientation_target = np.arcsin(sin_theta) # in range [-pi/2, pi/2]
                orientation_target_new = np.where(sin_theta > 0, np.where(cos_theta>0, orientation_target, np.pi - orientation_target), \
                    np.where(cos_theta > 0, np.pi - orientation_target, orientation_target + np.pi * 2))
                # if sin_theta > 0 and cos_theta > 0:    # first quadrant
                #     orientation_target_new = orientation_target
                # elif sin_theta > 0 and cos_theta <= 0: # second quadrant
                #     orientation_target_new = np.pi - orientation_target
                # elif sin_theta <= 0 and cos_theta > 0: # third quadrant
                #     orientation_target_new = np.pi - orientation_target 
                # else:                                  # forth quadrant
                #     orientation_target_new = orientation_target + np.pi * 2
                delta = np.abs((orientation_target_new - orientation_z))
                heading_rate = np.clip(delta, -const_heading_rate * self.sim_dt_, const_heading_rate * self.sim_dt_) / self.sim_dt_
                evader_action = np.concatenate([velocity_x / 7.0, velocity_y / 7.0, velocity_z / 7.0, heading_rate / 3.14], axis=1)
                if self.debug: print("Evader Speed:", evader_action)
                return evader_action


if __name__ == "__main__":
    env = NvDefault1(1, 1, False)

    obs = env.reset()
    for i in range(100):
        time.sleep(1)
        #action_interface = [action_space.sample() for action_space in env.action_space]
        action_interface = [np.array([0,0,0,0]), np.array([0,0,1,0]), np.array([0,0,0,0])]
        #action_interface = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        action_interface = np.expand_dims(np.concatenate(action_interface), axis=0)
        nxt, rew, don, inf = env.step(action_interface)
        #inf = inf[0]
        print("Step:", i, nxt[0][0][0:3],nxt[0][1][0:3], nxt[0][2][0:3], don, rew)
        for k in env.role_keys:
            if 'evader' in k:
                obs = env.obs[k]['proprioceptive'][0]['obs'].transpose()
                position = obs[env.map['position']['inx']].transpose()
        print("evader's p:", position)
        #print(env.obs)
        #print(obs[0])
        #print(nxt)
        #print(env.observation_space[0])
        #print("act_space=", env.action_space)
        #if don[0].all():
        #    print(don)
        #    break
         


