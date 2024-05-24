import numpy as np
from onpolicy.envs.uav.scenario import BaseScenarioUAV
from flightgym import MultiQuadrotorPIDVelCtlEnv_v0
from flightgym import MapStringFloatVector, MapStringFloat
from gym import spaces
import yaml
import copy
import time


class SimpleBase(BaseScenarioUAV):
    def __init__(self, num_envs, num_threads, render, cfg_path=None):
        super().__init__()
        if cfg_path is None:
            cfg_path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/envs/uav/scenarios/scenario.yaml'
        self.cfg_ = yaml.load(open(cfg_path, 'r'), Loader=yaml.FullLoader)

        # TODO: coordinate num_player
        self.env_ = MultiQuadrotorPIDVelCtlEnv_v0(num_envs, num_threads, render)
        self.num_env_ = num_env_ = self.env_.getNumOfEnvs()
        self.act_dim_ = act_dim_ = self.env_.getActDim()
        self.obs_dim_ = obs_dim_ = self.env_.getObsDim()
        self.num_agent_ = num_agent_ = self.env_.getNumAgent()
        self.sim_dt_ = sim_dt_ = self.env_.getSimTimeStep()
        self.episode_length = episode_length = self.env_.getEpisodeLength()
        print("Episode length is: ", self.episode_length)
        self.env_.connectUnity()

        print("[Checking Dims] act_dim={}, obs_dim={}, num_agent={}".format(act_dim_, obs_dim_, num_agent_))

        self.num_player_, self.num_pursuer_, self.num_evader_ = self._get_player()

        self.role_keys = ['pursuer_' + str(i) for i in range(self.num_pursuer_)] + ['evader_' + str(i) for i in range(self.num_evader_)]
        obs_arch_vals = [None for _ in range(self.num_player_)]
        obs_arch = dict(zip(self.role_keys, obs_arch_vals))

        self.obs_content = {'multi_agent_id': None, 'obs': None}
        for k in self.role_keys:
            if 'pursuer' in k:
                obs_arch[k] = {
                    'proprioceptive': [copy.deepcopy(self.obs_content)], 
                    'exterprioceptive': {
                        'teammate': [copy.deepcopy(self.obs_content) for _ in range(self.num_pursuer_-1)], 
                        'opponent': [copy.deepcopy(self.obs_content) for _ in range(self.num_evader_)], 
                        'context': []}}
            else:
                obs_arch[k] = {
                    'proprioceptive': [copy.deepcopy(self.obs_content)], 
                    'exterprioceptive': {
                        'teammate': [copy.deepcopy(self.obs_content) for _ in range(self.num_evader_-1)], 
                        'opponent': [copy.deepcopy(self.obs_content) for _ in range(self.num_pursuer_)], 
                        'context': []}}
        self.obs_arch = obs_arch

        print("[Checking Players] num_player={}, num_pursuer={}, num_evader={}".format(self.num_player_, self.num_pursuer_, self.num_evader_))

        # Define scenario-spacefic observation space; Currently does not support user-defined action space
        self.action_space, self.observation_space, self.observation_inx = self._get_space()
        # print("observation index:", self.observation_inx)
        # Initialize C++ environment data interface
        self.obs_interface = np.zeros((num_env_, obs_dim_), dtype=np.float32)
        self.act_interface = np.zeros((num_env_, act_dim_), dtype=np.float32)       
        self.reward_interface = np.zeros(shape=(num_env_, num_agent_), dtype=np.float32)
        self.done_interface = np.zeros(shape=(num_env_, num_agent_), dtype=np.bool)
        self.info_interface = MapStringFloatVector([MapStringFloat() for _ in range(num_env_)])
        self.last_obs_interface = np.zeros((num_env_, obs_dim_), dtype=np.float32)

    def _get_player(self):
        roles = [1 if r == 'pursuer' else 0 for r in self.cfg_['role']]
        num_pursuer = sum(roles)
        num_evader = len(roles) - num_pursuer
        assert num_evader <= 1, "Currently only support no more than one evader is supported!"

        return len(roles), num_pursuer, num_evader

    def _get_space(self):
        # obs_interface stores obs in the following format: [[obs_of_agent0 | obs_of_agent1 | obs_of_agent2 | ......], [obs_of_agent0 | obs_of_agent1 | obs_of_agent2 | ......].....], 
        # we hope to re-arrange each agent's obs in the following format: [proprioceptive, friend0, frient1 ... enemy0, enemy1 ...]

        # TODO: add shared environment observation; C++ already support
        # here we made an assumption that each player have the same action dim
        keys = ['pursuer_' + str(i) for i in range(self.num_pursuer_)] + ['evader_' + str(i) for i in range(self.num_evader_)]
        action_space = [spaces.Box(low=-1.0, high=1.0, shape=(self.act_dim_ // self.num_player_,), dtype=np.float32) for _ in range(self.num_player_)] 
        observation_space = [None for _ in keys]
        action_space = dict(zip(keys, action_space))
        observation_space = dict(zip(keys, observation_space))

        # get observation names
        pursuer_observation = self.cfg_['observation']['pursuer']
        evader_observation = self.cfg_['observation']['evader']

        detail_shape_dict = {'proprio_shape': None, 'teammate_shape': None, 'opponent_shape': None}

        self.sub_role_shape = dict()
        self.sub_role_shape['pursuer'] = copy.deepcopy(detail_shape_dict)
        self.sub_role_shape['evader'] = copy.deepcopy(detail_shape_dict)

        # get the corresponding obs's inx
        inxs_pursuer_proprioceptive = np.concatenate([np.array(self.map[pur_obs]['inx']) for pur_obs in pursuer_observation['proprioceptive']])
        inxs_pursuer_friend_exterprioceptive = np.concatenate([np.array(self.map[pur_obs]['inx']) for pur_obs in pursuer_observation['exterprioceptive']['friend']])
        inxs_pursuer_enemy_exterprioceptive = np.concatenate([np.array(self.map[pur_obs]['inx']) for pur_obs in pursuer_observation['exterprioceptive']['enemy']])

        self.sub_role_shape['pursuer']['proprio_shape'] = len(inxs_pursuer_proprioceptive)
        self.sub_role_shape['pursuer']['teammate_shape'] = len(inxs_pursuer_friend_exterprioceptive)
        self.sub_role_shape['pursuer']['opponent_shape'] = len(inxs_pursuer_enemy_exterprioceptive)

        inxs_evader_proprioceptive = np.concatenate([np.array(self.map[eva_obs]['inx']) for eva_obs in evader_observation['proprioceptive']])
        inxs_evader_friend_exterprioceptive = np.concatenate([np.array(self.map[eva_obs]['inx']) for eva_obs in evader_observation['exterprioceptive']['friend']])
        inxs_evader_enemy_exterprioceptive = np.concatenate([np.array(self.map[eva_obs]['inx']) for eva_obs in evader_observation['exterprioceptive']['enemy']])

        self.sub_role_shape['evader']['proprio_shape'] = len(inxs_evader_proprioceptive)
        self.sub_role_shape['evader']['teammate_shape'] = len(inxs_evader_friend_exterprioceptive)
        self.sub_role_shape['evader']['opponent_shape'] = len(inxs_evader_enemy_exterprioceptive)

        single_player_proprioceptive_obs = sum([self.map[k]['dim'] for k in self.map.keys()])
        single_player_original_full_obs = single_player_proprioceptive_obs * self.num_player_
        observation_inx = copy.deepcopy(self.obs_arch)
        for i in range(self.num_player_):
            
            history_inx = i * single_player_original_full_obs
            single_player_obs_dim = 0
            # get obs index for each pursuer
            if i < self.num_pursuer_:
                single_player_obs_inx = observation_inx['pursuer_' + str(i)]
                for j in range(self.num_player_):
                    # get player j's obs
                    if j == i:
                        single_player_obs_inx['proprioceptive'][0]['multi_agent_id'] = j
                        single_player_obs_inx['proprioceptive'][0]['obs'] = list(inxs_pursuer_proprioceptive + history_inx)
                        single_player_obs_dim += len(inxs_pursuer_proprioceptive)
                    else:
                        if j < self.num_pursuer_:
                            single_player_obs_inx['exterprioceptive']['teammate'][j if j < i else j-1]['multi_agent_id'] = j
                            single_player_obs_inx['exterprioceptive']['teammate'][j if j < i else j-1]['obs'] = list(inxs_pursuer_friend_exterprioceptive + history_inx)
                            single_player_obs_dim += len(inxs_pursuer_friend_exterprioceptive)
                        else:
                            single_player_obs_inx['exterprioceptive']['opponent'][j-self.num_pursuer_]['multi_agent_id'] = j
                            single_player_obs_inx['exterprioceptive']['opponent'][j-self.num_pursuer_]['obs'] = list(inxs_pursuer_enemy_exterprioceptive + history_inx)
                            single_player_obs_dim += len(inxs_pursuer_enemy_exterprioceptive)
                    history_inx += single_player_proprioceptive_obs

                observation_inx['pursuer_' + str(i)] = single_player_obs_inx
                observation_space['pursuer_' + str(i)] = spaces.Box(low=-1.0, high=1.0, shape=(single_player_obs_dim,), dtype=np.float32)

            # get obs index for each evader 
            else:
                single_player_obs_inx = observation_inx['evader_' + str(i-self.num_pursuer_)]
                for j in range(self.num_player_):
                    if j == i:
                        single_player_obs_inx['proprioceptive'][0]['multi_agent_id'] = j
                        single_player_obs_inx['proprioceptive'][0]['obs'] = list(inxs_evader_proprioceptive + history_inx)
                        single_player_obs_dim += len(inxs_evader_proprioceptive)
                    else:
                        if j < self.num_pursuer_:
                            single_player_obs_inx['exterprioceptive']['opponent'][j]['multi_agent_id'] = j
                            single_player_obs_inx['exterprioceptive']['opponent'][j]['obs'] = list(inxs_evader_enemy_exterprioceptive + history_inx)
                            single_player_obs_dim += len(inxs_evader_enemy_exterprioceptive)
                        else:
                            single_player_obs_inx['exterprioceptive']['teammate'][j-self.num_pursuer_ if j<i else j-self.num_pursuer_-1]['multi_agent_id'] = j
                            single_player_obs_inx['exterprioceptive']['teammate'][j-self.num_pursuer_ if j<i else j-self.num_pursuer_-1]['obs'] = list(inxs_evader_friend_exterprioceptive + history_inx)
                            single_player_obs_dim += len(inxs_evader_friend_exterprioceptive)
                    history_inx += single_player_proprioceptive_obs

                observation_inx['evader_' + str(i-self.num_pursuer_)] = single_player_obs_inx
                observation_space['evader_' + str(i-self.num_pursuer_)] = spaces.Box(low=-1.0, high=1.0, shape=(single_player_obs_dim,), dtype=np.float32)
        return action_space, observation_space, observation_inx

    def _set_action(self, act):
        act = act.astype(np.float32)
        assert act.shape == self.act_interface.shape, "Action dimention mismatch!"
        return act 

    def get_observation(self):
        # TODO: currently we assume that each agent can obtain the real-time observations of other agents, and there are no delay
        # obs_interface is a matrix with shape num_envs*obs_dim, where obs_dim iscludes all agents's observations, whose locations are specified by the self.observation_inx interface
        obs = copy.deepcopy(self.obs_arch)
        obs_interface = self.obs_interface.transpose()
        for player in obs.keys():
            # get proprioceptive observation
            obs[player]['proprioceptive'][0]['obs'] = obs_interface[self.observation_inx[player]['proprioceptive'][0]['obs']].transpose()
            # get exterprioceptive observation
            # 1. get teammate obs
            for i in range(len(obs[player]['exterprioceptive']['teammate'])):
                obs[player]['exterprioceptive']['teammate'][i]['obs'] = obs_interface[self.observation_inx[player]['exterprioceptive']['teammate'][i]['obs']].transpose()
            # 2. get opponent obs
            for i in range(len(obs[player]['exterprioceptive']['opponent'])):
                obs[player]['exterprioceptive']['opponent'][i]['obs'] = obs_interface[self.observation_inx[player]['exterprioceptive']['opponent'][i]['obs']].transpose()
            # 3. get context obs
            # TODO: add context obs in the future
        return obs

    def get_reward(self):
        rewards = np.split(self.reward_interface, self.num_player_, axis=1)
        rewards = dict(zip(self.role_keys, rewards))
        return rewards
    
    def get_done(self):
        dones = np.split(self.done_interface, self.num_player_, axis=1)
        dones = dict(zip(self.role_keys, dones))
        return dones
    
    def get_info(self):
        infos = [dict(info) for info in self.info_interface]
        return infos
    
    def step(self, act):
        act = self._set_action(act)
        self.env_.step(act, self.obs_interface, self.reward_interface, self.done_interface, self.info_interface, self.last_obs_interface)
        return self.get_observation(), self.get_reward(), self.get_done(), self.get_info()
    
    def reset(self):
        self.env_.reset(self.obs_interface)
        return self.get_observation() 
    


if __name__ == "__main__":
    env = SimpleBase(1, 1, False)
    act_interface = np.zeros((env.num_env_, env.act_dim_), dtype=np.float32)       

    obs = env.reset()
    for i in range(1):
        obs, rew, doe, info = env.step(act_interface)
        #print(doe, info[0])
        print(env.obs_interface)
