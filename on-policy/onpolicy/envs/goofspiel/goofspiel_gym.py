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
from onpolicy.envs.goofspiel.goofspiel_policy import Policy_goofspiel_random

import copy
import time
import random

def _t2n(x):
    return x.detach().cpu().numpy()

class goofspiel_symmetry:
    def __init__(self, num_threads, num_cards = 5, oppo_policy = None):
        self.standard_game = pyspiel.load_game("goofspiel(num_cards={},players=2)".format(num_cards))
        self.num_cards = num_cards
        print("num_cards = ", num_cards)

        observation_length = self.standard_game.information_state_tensor_size()
        act_dim = self.standard_game.num_distinct_actions()

        self.num_threads = num_threads

        self.game_list = []
        for i in range(self.num_threads):
            self.game_list.append(pyspiel.load_game("goofspiel(num_cards={},players=2)".format(num_cards)))

        self.states = [None for _ in range(self.num_threads)]

        self.episode_length = self.standard_game.max_game_length() - 1
        print("env name = goofspiel!")
        print("episode_length = ", self.episode_length)

        self.role_flags = np.zeros(self.num_threads)

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
        # print("action_dim = ", self.action_space[0].n)

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
        s_all, a_acts = self.collect_state(True)
        self.obs = s_all
        self.legal_act = a_acts
        
        self.obs = np.array(self.obs)
        self.legal_act = np.array(self.legal_act)

        # print("reset player = ", self.collect_current_player() - self.role_flags)

        return copy.deepcopy(self.obs[:, np.newaxis, :]), copy.deepcopy(self.legal_act[:, np.newaxis, :])
    
    def collect_state(self, master = True, mask = None):
        if mask is None:
            mask = np.ones(self.num_threads, dtype=bool)
        s_all = np.zeros((self.num_threads, self.obs_dim))
        a_acts_all = np.zeros((self.num_threads, self.act_dim), dtype=int)
        for i in range(self.num_threads):
            if mask[i]:
                if master:
                    s_all[i] = self.states[i].information_state_tensor(self.role_flags[i])
                    legal_actions = self.states[i].legal_actions(self.role_flags[i])
                    a_acts_all[i] = self.legal2available(legal_actions)[0]
                else:
                    s_all[i] = self.states[i].information_state_tensor(1 - self.role_flags[i])
                    legal_actions = self.states[i].legal_actions(1 - self.role_flags[i])
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

    def cat_actions(self, action_master, action_oppo):
        actions_all = np.zeros((self.num_threads, 2), dtype=int)
        for i in range(self.num_threads):
            actions_all[i][self.role_flags[i]] = action_master[i]
            actions_all[i][1 - self.role_flags[i]] = action_oppo[i]

        return actions_all

    def apply_action_all(self, actions_all):
        for i in range(self.num_threads):
            # print(actions_all[i])
            self.states[i].apply_actions([actions_all[i][0], actions_all[i][1]])
    
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
        s_all, a_acts = self.collect_state(False)
        oppo_action = self.get_oppo_action(s_all[:,np.newaxis,:], a_acts[:,np.newaxis,:])
        actions_all = self.cat_actions(np.concatenate(action_p+0.1).astype(int), np.concatenate(oppo_action+0.1).astype(int))

        self.apply_action_all(actions_all)
        
        self.across_chance_node()

        self.dones = self.check_terminal_state()[:, np.newaxis]
        self.rewards = self.get_rewards(self.dones)
        s_all, a_acts = self.collect_state(True)
        self.obs = s_all
        self.legal_act = a_acts

    def is_terminal_all(self):
        mask_terminal = copy.deepcopy(self.dones[:, 0])
        for i in range(self.num_threads):
            if mask_terminal[i]:
                self.role_flags[i] = np.random.randint(2)
                self.states[i] = self.game_list[i].new_initial_state()
        self.across_chance_node(mask_terminal)
        s_all, a_acts = self.collect_state(True, mask_terminal)
        self.obs[mask_terminal] = s_all[mask_terminal]
        self.legal_act[mask_terminal] = a_acts[mask_terminal]


    def step(self, actions):
        self.update_obs_all(actions)
        if self.oppo_policy is not None:
            self.oppo_rnn_states[self.dones == True] = np.zeros(((self.dones == True).sum(), self.oppo_policy.actor._recurrent_N, self.oppo_policy.actor.hidden_size), dtype=np.float32)
            self.oppo_masks = np.ones((self.num_threads, 1, 1), dtype=np.float32)
            self.oppo_masks[self.dones == True] = np.zeros(((self.dones == True).sum(), 1), dtype=np.float32)
        self.is_terminal_all()
        return copy.deepcopy(self.obs[:, np.newaxis, :]), copy.deepcopy(self.rewards[:,:,np.newaxis]), copy.deepcopy(self.dones), copy.deepcopy(self.infos), copy.deepcopy(self.legal_act[:,np.newaxis,:])
    

    def legal2available(self, legal_action):
        available_actions = np.zeros(self.action_space[0].n, dtype=int)
        available_actions[legal_action] = 1
        available_actions = available_actions[np.newaxis, :]

        return available_actions

    @torch.no_grad()
    def get_oppo_action(self, oppo_obs, available_actions):
        self.oppo_policy.actor.eval()
        oppo_action, oppo_rnn_states = self.oppo_policy.act(np.concatenate(oppo_obs),
                                                np.concatenate(self.oppo_rnn_states),
                                                np.concatenate(self.oppo_masks),
                                                np.concatenate(available_actions),
                                                deterministic=True)
        oppo_action = np.array(np.split(_t2n(oppo_action), self.num_threads))
        self.oppo_rnn_states = np.array(np.split(_t2n(oppo_rnn_states), self.num_threads))
        oppo_actions_env = np.concatenate([oppo_action[:, idx, :] for idx in range(1)], axis=1)

        return oppo_actions_env
    

    
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
    n_rollout = 200
    parser = get_config()
    parser = parse_args(parser)
    args=parser.parse_args()
    args.hidden_size = 128
    args.use_mixer = False
    num_cards = 5
    env_test = goofspiel_symmetry(n_rollout, num_cards)
    oppo_policy = Policy(args,
                        env_test.oppo_obs_space[0],
                        env_test.oppo_obs_space[0],
                        env_test.oppo_act_space[0])
    rule_policy = Policy_goofspiel_random(oppo_policy)
    env_test.oppo_policy = rule_policy
    obs, legal_actions = env_test.reset()
    print(obs)
    print(legal_actions)
    max_steps = np.zeros(n_rollout, dtype=int)
    max_steps[:] = -1
    overall_done = np.zeros((n_rollout,1), dtype=bool)
    print(env_test.role_flags)
    for i in range(env_test.episode_length):
        actions = np.zeros((n_rollout,1), dtype=int)
        for j in range(n_rollout):
            a_acts = np.arange(env_test.act_dim)
            a_acts = a_acts[legal_actions[j][0] == 1]
            actions[j][0] = np.random.choice(a_acts)
        # print(actions)
        s, r, d, infos, legal_actions = env_test.step(actions)

        mask_step = max_steps < 0

        max_steps[np.logical_and(mask_step, d[:,0])] = i

        overall_done = np.logical_or(overall_done, d)

        # print("Step: ", i)
        # print("obs =", s)
        # print("rewards = ", r)
        # print("dones = ", d)

    print("max_steps = ", max_steps)
    print(np.all(overall_done))
    print([np.min(max_steps), np.max(max_steps)])
        