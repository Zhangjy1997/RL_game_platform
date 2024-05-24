import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pyspiel
from open_spiel.python.algorithms import exploitability
from open_spiel.python.algorithms import policy_aggregator
from open_spiel.python import policy as openspiel_policy
from gym import spaces

import copy
import time
import random


def legal2available(n, legal_action):
    available_actions = np.zeros(n, dtype=int)
    available_actions[legal_action] = 1
    available_actions = available_actions[np.newaxis, :]

    return available_actions

class policy_like_spiel(openspiel_policy.Policy):
    def __init__(self, game, player_ids, policy_network, random_policy = False):
        super().__init__(game, player_ids)
        self.policy_network = policy_network
        self.act_dim = self.game.num_distinct_actions()
        self.random_policy = random_policy
    
    def action_probabilities(self, state, player_id = None):
        if player_id is None:
            foucs_player = state.current_player()
        else:
            foucs_player = player_id
        s = state.information_state_tensor(foucs_player)
        legal_actions = state.legal_actions()
        obs = np.array(s)[np.newaxis, :]
        a_act = legal2available(self.act_dim, legal_actions)
        rnn_states = np.zeros((1, self.policy_network.actor._recurrent_N, self.policy_network.actor.hidden_size), dtype=np.float32)
        masks = np.ones((1, 1), dtype=np.float32)
        act_state = self.policy_network.act(obs, rnn_states, masks, available_actions=a_act, deterministic=True)
        act_max = act_state[0][0]
        act_prob_dict = dict()
        if self.random_policy:
            for i in range(self.act_dim):
                if i in legal_actions:
                    act_prob_dict[i] = 1.0 / len(legal_actions)
                # else:
                #     act_prob_dict[i] = 0.0
        else:
            for i in range(self.act_dim):
                if i == act_max:
                    act_prob_dict[i] = 1.0
                elif i in legal_actions:
                    act_prob_dict[i] = 0.0

        return act_prob_dict
    
def calc_exp(game, policy):
    exp, expl_per_player = exploitability.nash_conv(
        game, policy, return_only_nash_conv=False)
    return np.array(exp / 2), expl_per_player

def gen_mix_spiel_policy(game, player_id, policies, probs):
    aggregator = policy_aggregator.PolicyAggregator(game)
    aggr_policies = aggregator.aggregate(player_id, policies, probs)
    return aggr_policies