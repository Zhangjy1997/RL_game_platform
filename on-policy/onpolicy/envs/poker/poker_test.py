import argparse
import copy
from collections import namedtuple
import pickle as pkl
import os
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_aggregator

np.set_printoptions(precision=4)
Transition = namedtuple('Transition', ['state', 'action', "actions_prob", 'a_log_prob', 'reward', 'next_state', 'done', "legal_action"])

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _state_values(state, num_players, policy):
    """Value of a state for every player given a policy."""
    if state.is_terminal():
        return np.array(state.returns())
    else:
        p_action = (
            state.chance_outcomes() if state.is_chance_node() else
            policy.action_probabilities(state).items())
    return sum(prob * _state_values(state.child(action), num_players, policy)
               for action, prob in p_action)

def kl_divergence(prob_a, prob_b):
    
    ori_shape = prob_b.shape
    prob_a = prob_a.reshape(-1)
    prob_b = prob_b.reshape(-1)
    prob_b[prob_a==0] = - np.inf
    prob_a = prob_a.reshape(ori_shape)
    prob_b = prob_b.reshape(ori_shape)

    prob_b = torch.softmax(prob_b, -1)
    res = (prob_a * torch.log(prob_a/prob_b))
    res = res.reshape(-1)
    prob_a = prob_a.reshape(-1)
    res[prob_a==0] = 0
    res = res.reshape(ori_shape)

    return res.sum(1)
    
def simulate_one(algorithm, div_weight, game, oppo_policies, oppo_id_pools, main_agent, buffer_to_add):
    n_steps_rec = [0]
    for oppo_idx in oppo_id_pools:
        # print(oppo_idx)
        oppo_policy = oppo_policies[oppo_idx]
        # train once
        state = game.new_initial_state()
        obs_list = []
        act_index_list = []
        act_prob_list = []
        legal_act_list = []
        act_prob_all_list = []
        legal_action_list = []

        player_train = int(np.random.uniform() < 0.5)
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            cur_player = state.current_player()
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                action_list, prob_list = zip(*outcomes_with_probs)
                action = np.random.choice(action_list, p=prob_list)
                state.apply_action(action)
            else:
                s = state.information_state_tensor(cur_player)
                if cur_player == player_train:
                    act_prob_all, action, act_prob = main_agent.select_action(
                        s, legal_actions)
                    obs_list.append(s)
                    act_index_list.append(action)
                    act_prob_list.append(act_prob)
                    legal_act_list.append(legal_actions)
                    act_prob_all_list.append(act_prob_all)
                    legal_act_onehot = torch.zeros_like(act_prob_all)
                    legal_act_onehot[legal_actions] = 1
                    legal_action_list.append(legal_act_onehot)

                else:
                    _, action, act_prob = oppo_policy.select_action(
                        s, legal_actions, False)

                state.apply_action(action)

        returns = state.returns()
        this_returns = returns[player_train]
        return_list = [this_returns]
        n_steps = len(act_index_list)
        n_steps_rec.append(n_steps_rec[-1]+n_steps)
        for _ in range(n_steps-1):
            # return_list.append(gamma * return_list[-1]) # discouted future reward
            return_list.append(0)
        return_list = return_list[::-1]
        for n in range(n_steps):
            if n == n_steps - 1:
                trans = Transition(obs_list[n], act_index_list[n], act_prob_all_list[n], act_prob_list[n], return_list[n], obs_list[n], 1, legal_action_list[n])
            else:
                trans = Transition(obs_list[n], act_index_list[n], act_prob_all_list[n], act_prob_list[n], return_list[n], obs_list[n+1], 0, legal_action_list[n])
            buffer_to_add.append(trans)
    return n_steps_rec


class PSD_PSRO_SOLVER(object):
    def __init__(self, args):
        # PSRO solver for symmetric game.
        self.game = pyspiel.load_game(f"{args.env}(players=2)")
        if args.env == "kuhn_poker":
            oracle_iters = 10000
            self.learn_step = 10
        elif args.env == "leduc_poker":
            oracle_iters = 20000
            self.learn_step = 100

        self.device = args.device
        self.env = args.env
        self.oracle_agent = args.oracle_agent
        self.sims_per_entry = args.sims_per_entry
        self.total_iters = args.total_iters
        self.oracle_iters = oracle_iters
        self.fsp_eps = 1e-3

        self.policy_set = [new_agent(self.game)]
        self.meta_game = np.zeros((1, 1))
        self.hist_ne = []

        # data record
        self.save_every = 50
        self.evaluate_every = 2

        self.algorithm = args.algorithm

        self.div_weight = args.div_weight

        DEBUG_ = False
        if DEBUG_:
            self.oracle_iters = 1000

    def fictitious_play(self, meta_game, init_strategy=None, max_iters=5000):
        exp = 1

         #if self.game_name.find("kuhn") > -1 else 5e-3
        if init_strategy is None:
            n = meta_game.shape[0]
            pop = np.random.uniform(0,1,(1,n))
            pop = pop/pop.sum(axis=1)[:,None]
        exps = []
        it_ = 0
        while abs(exp) > self.fsp_eps:
            average = np.mean(pop, axis=0)
            row_weighted_payouts = average.dot(meta_game)
            br = np.zeros_like(row_weighted_payouts)
            br[np.argmin(row_weighted_payouts)] = 1

            exp = average.dot(meta_game).dot(br.T)
            exps.append(exp)
            pop = np.vstack((pop, br))
            it_ += 1

            if it_ > max_iters:
                # print(f"FSP iterations: {it_}, Stop Iter, Exp: {exp}")
                break

        # print("FSP its: ", it_)
        return average, exps[-1]


    def sim_game(self, p1, p2, sims_per_entry=None):
        if sims_per_entry is None:
            sims_per_entry = self.sims_per_entry
        returns = []
        for _ in range(sims_per_entry):
            state = self.game.new_initial_state()
            agent1_playid = int(np.random.uniform() < 0.5)
            while not state.is_terminal():
                legal_actions = state.legal_actions()
                cur_player = state.current_player()
                if state.is_chance_node():
                    outcomes_with_probs = state.chance_outcomes()
                    action_list, prob_list = zip(*outcomes_with_probs)
                    action = np.random.choice(action_list, p=prob_list)
                    state.apply_action(action)
                else:
                    s = state.information_state_tensor(cur_player)
                    if cur_player == agent1_playid:
                        act, action, act_prob = p1.select_action(
                            s, legal_actions, noise=False)
                    else:
                        act, action, act_prob = p2.select_action(
                            s, legal_actions, noise=False)
                    state.apply_action(action)

            returns.append(state.returns()[agent1_playid])

        return np.mean(returns)

    def update_meta_game(self, new_policy):

        n = len(self.policy_set)
        new_row = np.zeros(n + 1)
        for idx in range(n):
            new_row[idx] = self.sim_game(new_policy, self.policy_set[idx])

        meta_game = np.zeros((n + 1, n + 1))
        meta_game[:n, :n] = self.meta_game
        meta_game[n, :] = new_row
        meta_game[:, n] = - new_row
        self.meta_game = meta_game

    def approximate_BR(self, oppo_policies, ne, cur_it, main_agent=None, iterations=None):

        # approximate best response
        if main_agent is None:
            main_agent = new_agent(self.game)
        if iterations is None:
            iterations = self.oracle_iters
        
        oppo_id_pools = np.random.choice(len(oppo_policies), p=ne, size=iterations)
        for idx in range(iterations//self.learn_step):
            buffer_to_add = []
            n_steps_rec = simulate_one(self.algorithm, self.div_weight, self.game,
                                       oppo_policies, oppo_id_pools[idx*self.learn_step:idx*self.learn_step+self.learn_step],
                                       main_agent, buffer_to_add)

            if self.algorithm == "psd_psro":
                states = torch.tensor([trans.state for trans in buffer_to_add]).to(self.device)
                action_probs = torch.stack([trans.actions_prob for trans in buffer_to_add], axis=0)
                kl_sum_all = []
                for oppo_pol in oppo_policies:
                    with torch.no_grad():
                        oppo_action_probs = oppo_pol.q_net(states)
                    kl_sum = kl_divergence(action_probs, oppo_action_probs) / self.learn_step
                    kl_sum_all.append(kl_sum)
                kl_sum_all = torch.stack(kl_sum_all, axis=0)
                min_idx = kl_sum_all.sum(1).argmin().item()
                for n_i in range(len(n_steps_rec)-1):
                    psd_score = kl_sum_all[min_idx,n_steps_rec[n_i]:n_steps_rec[n_i+1]].sum()
                    buffer_to_add[n_steps_rec[n_i+1]-1] = buffer_to_add[n_steps_rec[n_i+1]-1]._replace(
                        reward = buffer_to_add[n_steps_rec[n_i+1]-1].reward + 100 * psd_score * self.div_weight)

            for trans in buffer_to_add:
                main_agent.store_transition(trans)
            if self.algorithm == "psd_psro":
                main_agent.update(anchor=oppo_policies[min_idx], div_weight=self.div_weight)
            else:
                main_agent.update()

        return main_agent

    def add_new(self, policy):
        self.update_meta_game(policy)
        self.policy_set.append(policy)

    def update_ne_exp(self, it):
        self.ne, sub_exp = self.fictitious_play(self.meta_game)
        self.hist_ne.append(copy.deepcopy(self.ne))

        if it > 0 and (it % self.save_every == 0):
            model_dict_path = os.path.join("./model_logs", self.algorithm + str(args.seed) + "_div" + str(args.div_weight))
            if not os.path.exists(model_dict_path):
                os.makedirs(model_dict_path)

            model_pools = []
            for policy in self.policy_set: # TODO: youhua
                pol = copy.deepcopy(policy)
                pol.buffer = None
                model_pools.append(pol)
            res = {"models": model_pools,
                   "ne": self.hist_ne}
            save_path = os.path.join(model_dict_path, f"models_{it}.pkl")
            pkl.dump(res, open(save_path, "wb"))

        if (it % self.evaluate_every == 0):
            ne_thres = 0
            game = self.game
            if game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL:
                aggregator = policy_aggregator.PolicyAggregator(game)
                policies = [np.array(self.policy_set)[self.ne>ne_thres].tolist(), 
                            np.array(self.policy_set)[self.ne>ne_thres].tolist()]
                meta_stra = [self.ne[self.ne>ne_thres], self.ne[self.ne>ne_thres]]
                aggr_policies = aggregator.aggregate(range(2), policies, meta_stra)
            exp, expl_per_player = self.calc_exp(aggr_policies)

            print(f"Exploitability: {np.round(exp, 4)}, {expl_per_player}")

    def run(self):

        game = self.game
        main_policy = new_agent(game)
        it = 0
        start_it_time = time.time()
        while it < self.total_iters:
            clock_time = [time.time()]
            self.update_ne_exp(it)
            clock_time.append(time.time())
            self.policy_set.append(main_policy)
            clock_time.append(time.time())
            main_policy = self.approximate_BR(
                self.policy_set[:-1], self.ne, it, main_policy)
            clock_time.append(time.time())
            self.policy_set.pop()
            self.add_new(copy.deepcopy(main_policy))
            clock_time.append(time.time())
            it += 1
            print("=====Iter: %d finish, Duration: %.4f minutes" %
                  (it, (time.time() - start_it_time)/60))
            start_it_time = time.time()

    def calc_exp(self, policy):
        game = self.game
        exp, expl_per_player = exploitability.nash_conv(
            game, policy, return_only_nash_conv=False)
        return np.array(exp / 2), expl_per_player


if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(allow_abbrev=True)
    parser.add_argument("--env", default="leduc_poker")
    parser.add_argument("--algorithm", choices=["psro", "psd_psro"], default="psd_psro")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--sims_per_entry", type=int, default=1000)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--oracle_agent", type=str, default="DQN")
    parser.add_argument("--total_iters", type=int, default=162)
    parser.add_argument("--div_weight", type=float, default=1)

    args = parser.parse_args()
    args.device = args.device if torch.cuda.is_available() else "cpu"
    print(args)
    game = pyspiel.load_game(f"{args.env}(players=2)")
    print(game.num_distinct_actions())
    state = game.new_initial_state()
    if state.is_chance_node():
        outcomes_with_probs = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)
        action = np.random.choice(action_list, p=prob_list)
        state.apply_action(action)
    s = state.information_state_tensor(0)
    print("state of player_0 = ", s)
    s = state.information_state_tensor(1)
    print("state of player_1 = ", s)
    print("next step!")
    if state.is_chance_node():
        outcomes_with_probs = state.chance_outcomes()
        action_list, prob_list = zip(*outcomes_with_probs)
        action = np.random.choice(action_list, p=prob_list)
        print("action = ", action)
        state.apply_action(action)
    s = state.information_state_tensor(0)
    print("state of player_0 = ", s)
    s = state.information_state_tensor(1)
    print("state of player_1 = ", s)
    legal_actions = state.legal_actions()
    print("legal_act = ", legal_actions)
    act = np.zeros((1,1))
    state.apply_action(act[0])
    # print(s)
    print(state.information_state_string(0))
    print(state.information_state_string(1))
    args.OBS_DIM = game.information_state_tensor_size()
    args.ACTION_DIM = game.num_distinct_actions()
    args.HIDDEN_DIM = args.hidden_dim

    print(args.OBS_DIM)
    print(args.ACTION_DIM)
