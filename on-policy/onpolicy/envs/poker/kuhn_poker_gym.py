import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_aggregator
from gym import spaces
from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
from onpolicy.envs.poker.leduc_poker_policy import Policy_poker_random

import copy
import time
import random

def _t2n(x):
    return x.detach().cpu().numpy()

class kuhn_poker_symmetry:
    def __init__(self, num_threads, oppo_policy = None):
        self.standard_game = pyspiel.load_game("kuhn_poker(players=2)")

        observation_length = self.standard_game.information_state_tensor_size()
        act_dim = self.standard_game.num_distinct_actions()

        self.num_threads = num_threads

        self.game_list = []
        for i in range(self.num_threads):
            self.game_list.append(pyspiel.load_game("kuhn_poker(players=2)"))

        self.states = [None for _ in range(self.num_threads)]

        self.episode_length = (self.standard_game.max_game_length())
        print("env name = kuhn poker!")
        print("episode_length = ", self.episode_length)

        self.role_flags = np.zeros(self.num_threads)

        seed = int((int(time.time()*100) % 10000) * 1000)
        self.rng = np.random.default_rng(seed)

        self.dones = np.zeros((self.num_threads, 1), dtype=bool)
        self.rewards = np.zeros((self.num_threads, 1))
        self.obs = np.zeros((self.num_threads, observation_length))

        self.infos = []
        for i in range(self.num_threads):
            info = dict()
            self.infos.append(info)

        self.observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(observation_length,), dtype=np.float32) for _ in range(1)]
        self.share_observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(observation_length,), dtype=np.float32) for _ in range(1)]
        self.action_space = [spaces.Discrete(act_dim) for _ in range(1)]

        self.obs_dim = observation_length
        self.act_dim = act_dim

        self.oppo_obs_space = [spaces.Box(low=-1.0, high=1.0, shape=(observation_length,), dtype=np.float32) for _ in range(1)]
        self.oppo_act_space = [spaces.Discrete(act_dim) for _ in range(1)]

        if oppo_policy is None:
            self.oppo_policy = None
        else:
            self.oppo_policy = copy.deepcopy(oppo_policy)

    def reset(self):
        self.role_flags = np.random.randint(2, size=self.num_threads)
        self.states = []
        for i in range(self.num_threads):
            state = self.game_list[i].new_initial_state()
            self.states.append(state)

        if self.oppo_policy is not None:
            self.oppo_rnn_states = np.zeros((self.num_threads, 1, self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, 1, 1), dtype=np.float32)

        self.across_chance_node()
        cur_players = self.collect_current_player()
        mask_oppo_terminal =  np.equal(cur_players, 1 - self.role_flags)
        s_all, a_acts = self.collect_state()
        oppo_action = self.get_oppo_action_multi(s_all[:,np.newaxis,:], a_acts[:,np.newaxis,:], mask_oppo_terminal)
        self.apply_action_multi(oppo_action, mask_oppo_terminal)
        s_all, a_acts = self.collect_state()
        self.obs = s_all
        self.legal_act = a_acts
        
        self.obs = np.array(self.obs)
        self.legal_act = np.array(self.legal_act)

        # print("reset player = ", self.collect_current_player() - self.role_flags)
        # print("cur_players = ", self.collect_current_player())
        # print("roles = ", self.role_flags)
        # print("match = ", np.all(self.collect_current_player() == self.role_flags))

        return copy.deepcopy(self.obs[:, np.newaxis, :]), copy.deepcopy(self.legal_act[:, np.newaxis, :])
    
    def collect_state(self, mask = None):
        if mask is None:
            mask = np.ones(self.num_threads, dtype=bool)
        s_all = np.zeros((self.num_threads, self.obs_dim))
        a_acts_all = np.zeros((self.num_threads, self.act_dim), dtype=int)
        for i in range(self.num_threads):
            if mask[i]:
                cur_player = self.states[i].current_player()
                s_all[i] = self.states[i].information_state_tensor(cur_player)
                legal_actions = self.states[i].legal_actions()
                a_acts_all[i] = self.legal2available(legal_actions)[0]
        return s_all, a_acts_all
    
    def across_chance_node(self, mask = None):
        if mask is None:
            mask = np.ones(self.num_threads, dtype=bool)
        for i in range(self.num_threads):
            if mask[i]:
                while self.states[i].is_chance_node():
                    outcomes_with_probs = self.states[i].chance_outcomes()
                    action_list, prob_list = zip(*outcomes_with_probs)
                    action = np.random.choice(action_list, p=prob_list)
                    self.states[i].apply_action(action)

    def apply_action_multi(self, action_p, mask = None):
        if mask is None:
            mask = np.ones(self.num_threads, dtype=bool)
        
        for i in range(self.num_threads):
            if mask[i]:
                self.states[i].apply_action(action_p[i])

    def collect_current_player(self):
        cur_players = np.zeros((self.num_threads), dtype=int)
        for i in range(self.num_threads):
            cur_players[i] = self.states[i].current_player()

        return cur_players
    
    def check_terminal_state(self):
        mask = np.zeros((self.num_threads), dtype=bool)
        for i in range(self.num_threads):
            mask[i] = self.states[i].is_terminal()

        return mask
    
    def get_rewards(self, mask):
        rewards = np.zeros((self.num_threads, 1))
        for i in range(self.num_threads):
            if mask[i]:
                reward_ = self.states[i].returns()
                rewards[i] = reward_[self.role_flags[i]]

        return rewards

    def update_obs_all(self, action_p):
        # print("step player = ", self.collect_current_player(), self.role_flags)
        # print("dones = ", self.dones)
        s_all, a_acts = self.collect_state()
        self.apply_action_multi(action_p)
        
        cur_players = self.collect_current_player()
        # print("nxt player = ", cur_players , self.role_flags)
        mask_master_player = np.equal(cur_players, self.role_flags)
        mask_terminal = self.check_terminal_state()

        mask_ending = np.logical_or(mask_master_player, mask_terminal)

        while np.all(mask_ending) == False:
            self.across_chance_node()
            cur_players = self.collect_current_player()
            mask_player = cur_players >= 0
            mask_oppo_player = np.equal(cur_players, 1 - self.role_flags)
            s_all, a_acts = self.collect_state(mask_player)
            oppo_action = self.get_oppo_action_multi(s_all[:,np.newaxis,:], a_acts[:,np.newaxis,:], mask_oppo_player)
            self.apply_action_multi(oppo_action, mask_oppo_player)
            
            cur_players = self.collect_current_player()
            mask_master_player = np.equal(cur_players, self.role_flags)
            mask_terminal = self.check_terminal_state()
            mask_ending = np.logical_or(mask_master_player, mask_terminal)

        cur_players = self.collect_current_player()
        mask_player = cur_players >= 0
        self.dones = self.check_terminal_state()[:, np.newaxis]
        self.rewards = self.get_rewards(self.dones)
        s_all, a_acts = self.collect_state(mask_player)
        self.obs = s_all
        self.legal_act = a_acts

    def is_terminal_all(self):
        mask_terminal = copy.deepcopy(self.dones[:, 0])
        for i in range(self.num_threads):
            if mask_terminal[i]:
                self.role_flags[i] = np.random.randint(2)
                self.states[i] = self.game_list[i].new_initial_state()
        self.across_chance_node(mask_terminal)
        cur_players = self.collect_current_player()
        mask_oppo_player = np.equal(cur_players, 1 - self.role_flags)
        mask_oppo_terminal =  np.logical_and(mask_oppo_player, mask_terminal)
        s_all, a_acts = self.collect_state(mask_terminal)
        oppo_action = self.get_oppo_action_multi(s_all[:,np.newaxis,:], a_acts[:,np.newaxis,:], mask_oppo_terminal)
        self.apply_action_multi(oppo_action, mask_oppo_terminal)
        s_all, a_acts = self.collect_state(mask_terminal)
        self.obs[mask_terminal] = s_all[mask_terminal]
        self.legal_act[mask_terminal] = a_acts[mask_terminal]
            

    def reset_single(self, i):
        self.role_flags[i] = np.random.randint(2)
        self.states[i] = self.game_list[i].new_initial_state()
        while self.states[i].is_chance_node():
            outcomes_with_probs = self.states[i].chance_outcomes()
            action_list, prob_list = zip(*outcomes_with_probs)
            action = np.random.choice(action_list, p=prob_list)
            self.states[i].apply_action(action)

        cur_player = self.states[i].current_player()
        legal_actions = self.states[i].legal_actions()
        if cur_player == self.role_flags[i]:
            return self.states[i].information_state_tensor(cur_player), self.legal2available(legal_actions)[0]
        else:
            s = self.states[i].information_state_tensor(cur_player)
            if self.oppo_policy is None:
                oppo_action = np.random.choice(legal_actions)
            else:
                available_actions = self.legal2available(legal_actions)
                oppo_action = self.get_oppo_action_single(np.array(s)[np.newaxis, :], available_actions, i)
                oppo_action = oppo_action[0]
            self.states[i].apply_action(oppo_action)

        legal_actions = self.states[i].legal_actions()

        cur_player = self.states[i].current_player()
        if cur_player == self.role_flags[i]:
            return self.states[i].information_state_tensor(cur_player), self.legal2available(legal_actions)[0]

    def step(self, actions):
        self.update_obs_all(actions)
        if self.oppo_policy is not None:
            self.oppo_rnn_states[self.dones == True] = np.zeros(((self.dones == True).sum(), self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, 1, 1), dtype=np.float32)
            self.oppo_masks[self.dones == True] = np.zeros(((self.dones == True).sum(), 1), dtype=np.float32)
        self.is_terminal_all()
        return copy.deepcopy(self.obs[:, np.newaxis, :]), copy.deepcopy(self.rewards[:,:,np.newaxis]), copy.deepcopy(self.dones), copy.deepcopy(self.infos), copy.deepcopy(self.legal_act[:,np.newaxis,:])
    
    def update_obs(self, actions):
        for i in range(self.num_threads):
            obs, reward, done, legal_action = self.step_single(actions[i], i)
            self.obs[i] = obs
            self.rewards[i] = reward
            self.dones[i] = done
            self.legal_act[i] = legal_action

    def is_terminal(self):
        for i in range(self.num_threads):
            if self.dones[i]:
                s, legal_action = self.reset_single(i)
                self.obs[i] = s
                self.legal_act[i] = legal_action

    def step_single(self, action_p, i):
        cur_player = self.states[i].current_player()
        s = self.states[i].information_state_tensor(cur_player)
        legal_actions = self.states[i].legal_actions()
        self.states[i].apply_action(action_p[0])

        cur_player = self.states[i].current_player()
        rewards_i = 0
        done = False
        while cur_player != self.role_flags[i]:
            if self.states[i].is_terminal():
                rewards = self.states[i].returns()
                rewards_i = rewards[self.role_flags[i]]
                done = True
                return s, rewards_i, done, legal_actions
            if self.states[i].is_chance_node():
                outcomes_with_probs = self.states[i].chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                self.states[i].apply_action(action)
            elif cur_player == 1 - self.role_flags[i]:
                legal_actions = self.states[i].legal_actions()
                s = self.states[i].information_state_tensor(cur_player)
                if self.oppo_policy is None:
                    oppo_action = np.random.choice(legal_actions)
                else:
                    available_actions = self.legal2available(legal_actions)
                    oppo_action = self.get_oppo_action_single(np.array(s)[np.newaxis, :], available_actions, i)
                    oppo_action = oppo_action[0]
                self.states[i].apply_action(oppo_action)
            cur_player = self.states[i].current_player()

        s = self.states[i].information_state_tensor(cur_player)
        legal_actions = self.states[i].legal_actions()

        a_ac =  self.legal2available(legal_actions)[0]

        return s, rewards_i, done, a_ac

    
    def legal2available(self, legal_action):
        available_actions = np.zeros(self.action_space[0].n, dtype=int)
        available_actions[legal_action] = 1
        available_actions = available_actions[np.newaxis, :]

        return available_actions


    def _reset_action(self, act):
        assert act.shape[1] == 1 * self.action_space[0].n, 'wrong action dim!'
        # TODO: currently we randomly sample evader actions
        #evader_action = np.concatenate([np.expand_dims(np.concatenate([self.evader_action_space[i].sample() for i in range(self.num_evader_)]), axis=0) for _ in range(self.num_env_)], axis=0)
        if self.oppo_policy is None:
            oppo_action = np.concatenate([np.expand_dims(np.concatenate([self.oppo_act_space[i].sample() for i in range(1)]), axis=0) for _ in range(self.num_threads)], axis=0)
        else:
            oppo_action = self.get_oppo_action(self.oppo_obs)
        ## TODO: currently we clip the network action, in future ......
        #act = np.clip(act, -1, 1)
        #print(act, evader_action)
        action = np.concatenate([act, oppo_action], axis=1)
        return action
    
    @torch.no_grad()
    def get_oppo_action_single(self, oppo_obs, available_actions, i):
        self.oppo_policy.actor.eval()
        oppo_action, oppo_rnn_states = self.oppo_policy.act(oppo_obs,
                                                self.oppo_rnn_states[i],
                                                self.oppo_masks[i],
                                                available_actions,
                                                deterministic=True)
        self.oppo_rnn_states[i] = _t2n(oppo_rnn_states)

        return oppo_action
    
    @torch.no_grad()
    def get_oppo_action_multi(self, oppo_obs, available_actions, mask):
        self.oppo_policy.actor.eval()
        oppo_action, oppo_rnn_states = self.oppo_policy.act(np.concatenate(oppo_obs),
                                                np.concatenate(self.oppo_rnn_states),
                                                np.concatenate(self.oppo_masks),
                                                np.concatenate(available_actions),
                                                deterministic=False)
        oppo_rnn_states_ = np.array(np.split(_t2n(oppo_rnn_states), self.num_threads))
        self.oppo_rnn_states[mask] = oppo_rnn_states_[mask]

        return oppo_action
    
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
    env_test = kuhn_poker_symmetry(100)
    oppo_policy = Policy(args,
                        env_test.oppo_obs_space[0],
                        env_test.oppo_obs_space[0],
                        env_test.oppo_act_space[0])
    rule_policy = Policy_poker_random(oppo_policy)
    env_test.oppo_policy = rule_policy

    obs, legal_actions = env_test.reset()
    # print(obs)
    # print(legal_actions)
    for i in range(env_test.episode_length):
        actions = np.zeros((100,1), dtype=int)
        for j in range(100):
            a_acts = np.arange(env_test.act_dim)
            a_acts = a_acts[legal_actions[j][0] == 1]
            actions[j][0] = np.random.choice(a_acts)
        print(actions)
        s, r, d, infos, legal_actions = env_test.step(actions)

        # print("Step: ", i)
        # print("obs =", s)
        # print("rewards = ", r)
        # print("dones = ", d)
        