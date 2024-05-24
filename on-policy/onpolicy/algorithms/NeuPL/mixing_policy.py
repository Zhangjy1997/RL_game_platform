import torch
import numpy as np
from onpolicy.algorithms.utils.util import init, check

import time

def find_unique_rows(matrix, threshold=0.001):
    unique_rows = []
    unique_indices = []

    for i, row in enumerate(matrix):
        if np.sum(row)>0.5:
            if not unique_rows:
                unique_rows.append(row)
                unique_indices.append(i)
            else:
                is_unique = True
                for unique_row in unique_rows:
                    if np.linalg.norm(row - unique_row) < threshold:
                        is_unique = False
                        break

                if is_unique:
                    unique_rows.append(row)
                    unique_indices.append(i)

    return np.array(unique_rows), unique_indices

class Mixing_policy:
    def __init__(self, policies):
        self.policy_num = len(policies)
        self.policy_list  = policies
        self.probs = np.random.rand(self.policy_num)
        self.probs = self.probs / self.probs.sum()
        self.selected_index = np.random.choice(self.policy_num, p=self.probs)
        self.history_index = np.zeros(self.policy_num)

    def set_probs(self, probs):
        self.probs = probs
        self.probs /= sum(self.probs)
        self.history_index = np.zeros(self.policy_num)
        self.update_selected_index()

    def update_selected_index(self):
        self.selected_index = np.random.choice(self.policy_num, p=self.probs)
        self.history_index[self.selected_index] += 1

    def get_selected_index(self):
        return self.selected_index
    
    def act(self, obs, rnn_state, rnn_mask, deterministic = True):
        nxt_action, nxt_rnn_states = self.policy_list[self.selected_index].act(obs, rnn_state, rnn_mask, deterministic = deterministic)
        return nxt_action, nxt_rnn_states

# bridge class
class Parallel_mixing_policy:
    def __init__(self, num_threads, policies, probs = None, device = torch.device("cpu")):
        self.num_threads = num_threads
        self.policies = policies
        self.actor = policies[0].actor
        self.sub_select = np.empty(self.num_threads, dtype=int)
        self.sort_line = np.argsort(self.sub_select)
        self.anti_sort_line = np.empty_like(self.sort_line)
        self.num_policy = len(policies)
        self.tpdv = dict(dtype=torch.float32, device=device)
        for i in range(len(policies)):
            self.policies[i].actor.eval()
        
        self.mp_list = []
        for i in range(self.num_threads):
            self.mp_list.append(Mixing_policy(self.policies))

        if probs is not None:
            self.probs = probs
            self.set_probs_all(probs)
        else:
            self.get_sub_select()

    def set_sort_line(self):
        self.sort_line = np.argsort(self.sub_select)
        self.anti_sort_line[self.sort_line] = np.arange(len(self.sort_line))

    def get_sub_select(self):
        for i in range(self.num_threads):
            self.sub_select[i] = self.mp_list[i].get_selected_index()

        self.num_count = np.bincount(self.sub_select)
        count_len = len(self.num_count)
        if count_len < self.num_policy:
            self.num_count = np.pad(self.num_count ,(0,self.num_policy - count_len),'constant')
        self.set_sort_line()


    def expand_sort_line(self, group_size):
        offsets = np.arange(group_size)
        all_indices = self.sort_line[:, None] * group_size + offsets
        expand_indices = all_indices.flatten()

        inver_all_inx = np.empty_like(expand_indices)
        inver_all_inx[expand_indices] = np.arange(len(expand_indices))

        return expand_indices, inver_all_inx
        
    def set_probs_all(self, probs):
        self.probs = probs
        for i in range(self.num_threads):
            self.mp_list[i].set_probs(probs)
        self.get_sub_select()

    def set_probs_channel(self, probs, i):
        self.mp_list[i].set_probs(probs)
        self.get_sub_select()

    def set_probs_mat(self, probs_mat):
        for i in range(self.num_threads):
            self.mp_list[i].set_probs(probs_mat[i])
        self.get_sub_select()

    def set_probs_multi_channel(self, probs_mat, inx):
        for i in inx:
            self.mp_list[i].set_probs(probs_mat[i])
        self.get_sub_select()

    def update_index_all(self):
        for i in range(self.num_threads):
            self.mp_list[i].update_selected_index()
        self.get_sub_select()

    def update_index_channel(self, i):
        self.mp_list[i].update_selected_index()
        self.get_sub_select()

    def update_index_multi_channels(self, inx):
        for i in inx:
            self.mp_list[i].update_selected_index()
        self.get_sub_select()

    def act(self, obs, rnn_state, rnn_mask, available_actions=None, deterministic = True):
        group_size = len(obs) // self.num_threads
        ex_sort, inver_sort = self.expand_sort_line(group_size)
        obs_channels = obs[ex_sort]
        rnn_state_channels = rnn_state[ex_sort]
        rnn_mask_channels = rnn_mask[ex_sort]
        obs_channels = check(obs_channels).to(**self.tpdv)
        rnn_state_channels = check(rnn_state_channels).to(**self.tpdv)
        rnn_mask_channels = check(rnn_mask_channels).to(**self.tpdv)
        if available_actions is not None:
            available_actions_channels = available_actions[ex_sort]
            available_actions_channels = check(available_actions_channels).to(**self.tpdv)
            use_a_acts = True
        else:
            use_a_acts = False
        actions_ch = []
        rnn_state_ch = []
        inx_data = 0
        #print(self.num_count)
        for i in range(len(self.policies)):
            if self.num_count[i]>0:
                indices_data = range(inx_data, inx_data + group_size*self.num_count[i])
                action_temp, rnn_state_temp = self.policies[i].act(obs_channels[indices_data], 
                                                                   rnn_state_channels[indices_data], 
                                                                   rnn_mask_channels[indices_data], 
                                                                   available_actions=available_actions_channels[indices_data] if use_a_acts else None, 
                                                                   deterministic = deterministic)
                actions_ch.append(action_temp)
                rnn_state_ch.append(rnn_state_temp)
                inx_data +=group_size * self.num_count[i]
        action_out = torch.cat(actions_ch, dim=0)
        rnn_state_out = torch.cat(rnn_state_ch, dim=0)
        action_out = action_out[inver_sort]
        rnn_state_out = rnn_state_out[inver_sort]
        return action_out, rnn_state_out
    
class Mat_mixing_policy(Parallel_mixing_policy):
    def __init__(self, num_threads, policies, matrix):
        self.matrix = matrix
        self.effect_mat, self.effect_inx = find_unique_rows(matrix)
        print("effect_policy = ",self.effect_mat)
        print("effect_inx = ",self.effect_inx)
        print("effect_num = ",len(self.effect_inx))
        self.prob_inx = np.random.randint(0,len(self.effect_inx))
        super().__init__(num_threads, policies, probs=self.effect_mat[self.prob_inx,:])

    def set_mat(self,mat):
        self.matrix = mat
        self.effect_mat, self.effect_inx = find_unique_rows(mat)
        print("effect_policy = ",self.effect_mat)
        print("effect_inx = ",self.effect_inx)
        print("effect_num = ",len(self.effect_inx))
        self.prob_inx = np.random.randint(0,len(self.effect_inx))
        self.set_probs_all(self.effect_mat[self.prob_inx,:])
    
    def update_mat_inx(self):
        self.prob_inx = np.random.randint(0,len(self.effect_inx))
        self.set_probs_all(self.effect_mat[self.prob_inx,:])
        print("select policy {}, probs = {}".format(self.effect_inx[self.prob_inx], self.effect_mat[self.prob_inx,:]))
        return self.effect_inx[self.prob_inx], self.effect_mat[self.prob_inx,:]

# class sigma_mix_policy:
#     def __init__(self, n_threads, policies, prob_sigma):

#         self.n_threads = n_threads
#         self.policy_sigma = policies
#         self.probs = prob_sigma
#         self.id_sigma = identify_sigma
#         self.id_length = len(identify_sigma)
#         assert n_threads == len(prob_sigma), "wrong dimension!"
#         self.selected_index = np.zeros(n_threads)
#         self.selected_sigma = np.zeros_like(prob_sigma)
#         self.update_selection()

        
#     def update_selection(self, inx = None):
#         up_inx = range(self.n_threads) if inx is None else inx
#         for i in up_inx:
#             self.selected_index[i] = np.random.choice(self.policy_num, p=self.probs[i])
#             self.selected_sigma[i] = self.id_sigma[self.selected_index[i],:]
#         self.policy_sigma.set_sigma(self.selected_sigma)

#     def update_probs(self, prob_sigma):
#         self.probs = prob_sigma
#         self.updata_selection()

#     def act(self, obs, rnn_state, rnn_mask, deterministic = True):
#         action_nxt, rnn_state_nxt = self.policy_sigma.act(obs, rnn_state, rnn_mask, deterministic = deterministic)
#         return action_nxt, rnn_state_nxt
        

        

if __name__ == "__main__":
    num_thread = 32
    policies = [i for i in range(33)]
    # Policy_test = Mixing_policy(policies)
    # prob_test = np.array([0.0 + i  for i in range(3)])
    # Policy_test.set_probs(prob_test)
    prob_test = np.array([6.143e-16, 1.097e-15, 1.735e-15, 1.470e-18, 1.864e-15, 9.797e-16, 0.000e+00,
                 0.000e+00, 0.000e+00, 0.000e+00, 2.272e-15, 0.000e+00, 8.522e-16, 0.000e+00,
                 0.000e+00, 1.455e-02, 0.000e+00, 0.000e+00, 0.000e+00, 1.071e-01, 0.000e+00,
                 0.000e+00, 4.680e-02, 3.551e-01, 5.037e-02, 6.033e-02, 3.657e-01, 0.000e+00,
                 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00])
    eps = 2500
    steps = 200
    lambda_policy = np.random.uniform(1, 4, len(policies))
    probs_win = np.random.uniform(0, 1, len(policies))

    ture_payoff = np.dot(lambda_policy * (2* probs_win - 1), prob_test)

    sample_N = np.zeros(len(policies), dtype=int)
    inx_N = np.zeros(len(policies), dtype=int)
    # print("shape of delta = ", delta_T.shape)
    for i in range(len(policies)):
        sample_N[i] = np.random.poisson(lam=lambda_policy[i] * eps * len(policies), size=1)

    delta_T = np.zeros((len(policies), max(sample_N)), dtype= int)

    print("sample_N = ", sample_N)

    for i in range(len(policies)):
        integer_samples = np.random.randint(0, eps*steps * len(policies), size=sample_N[i])
        sorted_integer_samples = np.sort(integer_samples)
        start_i = 0
        for j in range(len(sorted_integer_samples)):
            delta_T[i][j] = sorted_integer_samples[j] - start_i
            start_i = sorted_integer_samples[j]

    prob_new = prob_test * lambda_policy / (sum(prob_test * lambda_policy))

    Parallel_Policy_test = Parallel_mixing_policy(num_thread, policies, probs=prob_new)

    next_done_time = np.zeros(num_thread, dtype= int)
    thread_steps = np.zeros(num_thread, dtype=int)
    policy_history = np.zeros(len(policies))
    for i in range(num_thread):
        policy_inx = Parallel_Policy_test.mp_list[i].get_selected_index()
        next_done_time[i] = delta_T[policy_inx][inx_N[policy_inx]]
        inx_N[policy_inx] += 1
        policy_history[policy_inx] += 1

    total_win = 0
    total_round = 0

    for i in range(eps*steps):
        thread_steps += 1
        for j in range(num_thread):
            if thread_steps[j] >= next_done_time[j]:
                prev_policy_inx = Parallel_Policy_test.mp_list[j].get_selected_index()
                win_state = np.random.binomial(1, probs_win[prev_policy_inx], 1)
                # print(win_state)
                lose_state = 1 - win_state[0]
                total_win += 2 * win_state[0] - 1
                Parallel_Policy_test.update_index_channel(j)
                policy_inx = Parallel_Policy_test.mp_list[j].get_selected_index()
                thread_steps[j] = 0
                next_done_time[j] = delta_T[policy_inx][inx_N[policy_inx]]
                inx_N[policy_inx] += 1
                policy_history[policy_inx] += 1

    probs_eval = policy_history/sum(policy_history)
    prob_test[prob_test < 1e-9] = 0
    print(probs_eval)
    print(prob_test)

    print("prob_ture = ", ture_payoff)
    print("prob_eval = ", total_win/(num_thread * eps))