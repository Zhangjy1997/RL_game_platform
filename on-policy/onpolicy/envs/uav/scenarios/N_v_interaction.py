import numpy as np
from onpolicy.envs.uav.scenarios.vec_env_wrapper import SimpleBase
from onpolicy.envs.uav.scenario import BaseScenarioUAV
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
from gym import spaces
from onpolicy.config import get_config
import yaml
import copy
import time
import torch

from onpolicy.utils.plot3d_test import plot_track
from onpolicy.algorithms.NeuPL.Policy_rule import Policy_E2P_suboptimal as evader_rule_policy
from onpolicy.algorithms.NeuPL.Policy_rule import Policy_P2E_straight as pursuer_rule_policy
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as multi_policy

def _t2n(x):
    return x.detach().cpu().numpy()

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

class NvInteraction(SimpleBase):
    def __init__(self, num_envs, num_threads, render, team_name, oppo_name, oppo_policy=None):
        super().__init__(num_envs, num_threads, render)
        self.team_name = team_name
        self.oppo_name = oppo_name
        ## for debugging only
        self.debug = False
        if self.debug:
            self.obs_log = []
            self.rollout_counter = 0
            self.step_counter = 0

        observation_space = copy.deepcopy(self.observation_space)
        oppo_obs_space = copy.deepcopy(self.observation_space)
        action_space = copy.deepcopy(self.action_space)
        oppo_act_space = copy.deepcopy(self.action_space)
        for k in self.role_keys:
            if self.oppo_name in k:
                observation_space.pop(k)
                action_space.pop(k)
            if self.team_name in k:
                oppo_obs_space.pop(k)
                oppo_act_space.pop(k)
        self.observation_space = list(observation_space.values())
        self.action_space = list(action_space.values())
        self.oppo_act_space = list(oppo_act_space.values())
        self.oppo_obs_space = list(oppo_obs_space.values())
        self.oppo_policy = copy.deepcopy(oppo_policy)
        self.obs = None
        self.num_team, self.num_oppo =self._get_team()
    
    def _get_team(self):
        roles = [1 if self.team_name in r else 0 for r in self.cfg_['role']]
        num_team = sum(roles)
        num_oppo = len(roles) - num_team

        return num_team, num_oppo
    
    def _reset_action(self, act):
        assert act.shape[1] == self.num_team * self.action_space[0].shape[0], 'wrong action dim!'
        # TODO: currently we randomly sample evader actions
        #evader_action = np.concatenate([np.expand_dims(np.concatenate([self.evader_action_space[i].sample() for i in range(self.num_evader_)]), axis=0) for _ in range(self.num_env_)], axis=0)
        if self.oppo_policy == None:
            evader_action = self.get_evader_action_rule(const_speed=3, const_heading_rate=0.57/2)
        else:
            evader_action = self.get_oppo_action(dict2vector(self.oppo_obs))
        ## TODO: currently we clip the network action, in future ......
        #act = np.clip(act, -1, 1)
        #print(act, evader_action)
        if self.team_name in self.role_keys[0]:
            action = np.concatenate([act, evader_action], axis=1)
        else:
            action = np.concatenate([evader_action, act], axis=1)
        return action

    def get_observation_new(self):
        obs = super().get_observation()
        self.obs = copy.deepcopy(obs)
        self.oppo_obs = copy.deepcopy(obs)
        if self.debug: tmp_obs = []
        for k in self.role_keys:
            if self.debug: tmp_obs.append(obs[k]['proprioceptive'][0]['obs'][0][:3])
            if self.oppo_name in k:
                obs.pop(k)
            if self.team_name in k:
                self.oppo_obs.pop(k)
        if self.debug:
            self.obs_log.append(np.concatenate(tmp_obs))
        return dict2vector(obs)

    def get_reward_new(self):
        reward = super().get_reward()
        for k in self.role_keys:
            if self.oppo_name in k:
                reward.pop(k)
        return np.expand_dims(np.concatenate(list(reward.values()), axis=1), axis=-1)
    
    def get_done_new(self):
        done = super().get_done()
        oppo_done = copy.deepcopy(done)
        for k in self.role_keys:
            if self.oppo_name in k:
                done.pop(k)
            if self.team_name in k:
                oppo_done.pop(k)
        self.oppo_done = np.concatenate(list(oppo_done.values()), axis=1)
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
        if self.oppo_policy is not None:
            self.oppo_rnn_states[self.oppo_done == True] = np.zeros(((self.oppo_done == True).sum(), self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_env_, self.num_oppo, 1), dtype=np.float32)
            self.oppo_masks[self.oppo_done == True] = np.zeros(((self.oppo_done == True).sum(), 1), dtype=np.float32)
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
        if self.oppo_policy is not None:
            self.oppo_rnn_states = np.zeros((self.num_env_, self.num_oppo, self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_env_, self.num_oppo, 1), dtype=np.float32)
        return self.get_observation_new()
    
    @torch.no_grad()
    def get_oppo_action(self, oppo_obs):
        self.oppo_policy.actor.eval()
        oppo_action, oppo_rnn_states = self.oppo_policy.act(np.concatenate(oppo_obs),
                                                np.concatenate(self.oppo_rnn_states),
                                                np.concatenate(self.oppo_masks),
                                                deterministic=True)
        self.oppo_action = np.array(np.split(_t2n(oppo_action), self.num_env_))
        self.oppo_rnn_states = np.array(np.split(_t2n(oppo_rnn_states), self.num_env_))
        oppo_actions_env = np.concatenate([self.oppo_action[:, idx, :] for idx in range(self.num_oppo)], axis=1)

        return oppo_actions_env
    
    def get_evader_action_rule(self, const_speed=None, const_heading_rate=None, random=False):
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
    

def restore(oppo_policy, model_dir, use_mixer = True):
    """Restore policy's networks from a saved model."""
    policy_actor_state_dict = torch.load(str(model_dir) + '/actor.pt')
    oppo_policy.actor.load_state_dict(policy_actor_state_dict)
    if use_mixer:
        policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer.pt')
        oppo_policy.mixer.load_state_dict(policy_mixer_state_dict)

def parse_args(parser):
    parser.add_argument("--scenario_name", type=str,
                        default="simple_uav", 
                        help="which scenario to run on.")
    parser.add_argument("--num_agents", type=int, default=4,
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
                        default=32, help="hidden size of encoder")
    parser.add_argument("--proprio_shape", type=int, default=13,
                        help="proprio_shape")
    parser.add_argument("--teammate_shape", type=int, default=7,
                        help="teammate")
    parser.add_argument("--opponent_shape", type=int, default=3,
                        help="opponent_shape")
    parser.add_argument("--n_head", type=int, default=4, help="n_head")
    parser.add_argument("--d_k", type=int, default= 8, help="d_k")
    parser.add_argument("--attn_size", type=int, default=32, help="attn_size")

    return parser


if __name__ == "__main__":
    parser = get_config()
    parser = parse_args(parser)
    args=parser.parse_args()
    args.hidden_size = 128
    args.use_mixer = False
    n_th = 32
    envs=NvInteraction(n_th, n_th, False, 'pursuer', 'evader')
    model_dir= "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/scripts/train_uav_scripts/wandb/run-20231109_011724-3ilxzbv9/files"
    oppo_policy = Policy(args,
                            envs.oppo_obs_space[0],
                            envs.oppo_obs_space[0],
                            envs.oppo_act_space[0])
    
    #restore(pursuer_policy, model_dir)
    print(args.hidden_size)
    #envs.oppo_policy = copy.deepcopy(pursuer_policy)
    #envs.oppo_policy = evader_rule_policy(oppo_policy, 3, 0.57/2)
    path = "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattacker.mat"
    samples = np.linspace(-105, 105, 51)
    policies = []
    policies.append(evader_rule_policy(oppo_policy, 7, 0.57/2,path, samples , 13, device=torch.device("cpu")))
    envs.oppo_policy = multi_policy(1, policies)
    obs = envs.reset()
    print("reset obs = ", obs)
    ob_op = dict2vector(envs.oppo_obs)

    team_track = []
    oppo_track = []

    team_track.append([obs[0][i][0:3] for i in range(envs.num_team)])
    oppo_track.append([ob_op[0][i][0:3] for i in range(envs.num_oppo)])

    print("sub_role_size = ", envs.sub_role_shape)

    start_time = time.time()
    game_long = 200
    for i in range(game_long):
        # action_interface = [np.array([0.5,0.5,0,0]), np.array([0.9,-0.2,0,0]), np.array([-0.5,0.5,0,0])]
        #action_interface = [np.array([0,0,0.5,0])]
        action_interface = np.concatenate([np.expand_dims(np.concatenate([envs.action_space[i].sample() for i in range(envs.num_team)]), axis=0) for _ in range(envs.num_env_)], axis=0)
        # action_interface = np.expand_dims(np.concatenate(action_interface), axis=0)
        nxt, rew, don, inf = envs.step(action_interface)
        # if don[0].all():
        #     break
        team_track.append([nxt[0][i][0:3] for i in range(envs.num_team)])
        ob_op = dict2vector(envs.oppo_obs)
        oppo_track.append([ob_op[0][i][0:3] for i in range(envs.num_oppo)])
        #print("Step:", i, nxt[0][0][0:3], don, rew)
        print("\nStep:", i)
        # # print("info = ", inf)
        print("reward = ", rew)
        # print("obs=", nxt)
        print("done=", don)
        # print(ob_op[0][0][0:3], nxt[0][0][0:3])
        # print(nxt[0][1][0:3])
        # print(nxt[0][2][0:3])
        # print("\n")

    end_time = time.time()

    print("fps = ", (game_long * n_th)/(end_time - start_time))
    #print(len(oppo_track[0]))
    #print(team_track[1][0], team_track[1][0][0])
    # plot_track(oppo_track, team_track)