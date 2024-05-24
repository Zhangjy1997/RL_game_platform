import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_share import R_Actor_share, R_Critic_share
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.order_moudle_sort import Order_Mixer
import numpy as np


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.use_mixer = args.use_mixer

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.policy_inx = 0
        self.population_size = args.population_size
        self.policy_num = self.population_size - 1

        if self.use_mixer:
            self.team_name = args.team_name
            self.mixer=Order_Mixer(args, self.team_name, self.device)
            self.actor = R_Actor_share(args, self.mixer.output_space(), self.act_space, self.device)
            self.critic = R_Critic_share(args, self.mixer.output_space(), self.device)
        else:
            self.actor = R_Actor_share(args, self.obs_space, self.act_space, self.device)
            self.critic = R_Critic_share(args, self.share_obs_space, self.device)

        if self.use_mixer:
            self.actor_optimizer = torch.optim.Adam([{'params': self.mixer.parameters()},
                                                     {'params': self.actor.parameters()}
                                                ], lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam([{'params': self.mixer.parameters()},
                                                     {'params': self.critic.parameters()}
                                                ],lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def set_policy_inx(self, inx):
        self.policy_inx = inx

    def set_id_sigma(self, indices):
        self.policy_indices = indices
        self.num_indices = len(self.policy_indices)
        self.get_sub_select()

    def set_sort_line(self):
        self.sort_line = np.argsort(self.policy_indices)
        self.anti_sort_line = np.empty_like(self.sort_line)
        self.anti_sort_line[self.sort_line] = np.arange(len(self.sort_line))

    def get_sub_select(self):
        self.num_count = np.bincount(self.policy_indices)
        count_len = len(self.num_count)
        if count_len < self.policy_num:
            self.num_count = np.pad(self.num_count ,(0,self.policy_num - count_len),'constant')
        self.set_sort_line()

    def expand_sort_line(self, group_size):
        offsets = np.arange(group_size)
        all_indices = self.sort_line[:, None] * group_size + offsets
        expand_indices = all_indices.flatten()

        inver_all_inx = np.empty_like(expand_indices)
        inver_all_inx[expand_indices] = np.arange(len(expand_indices))

        return expand_indices, inver_all_inx

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        if self.use_mixer:
            obs=self.mixer(obs)
            cent_obs=self.mixer(cent_obs)

        group_size = len(obs) // self.num_indices
        ex_sort, inver_sort = self.expand_sort_line(group_size)
        obs_ch = obs[ex_sort]
        cent_obs_ch = cent_obs[ex_sort]
        masks_ch = masks[ex_sort]
        rnn_states_actor_ch = rnn_states_actor[ex_sort]
        rnn_states_critic_ch = rnn_states_critic[ex_sort]
        if available_actions is not None:
            available_actions_ch = available_actions[ex_sort]
        else:
            available_actions_ch = None

        actions_ch = []
        action_log_probs_ch = []
        nxt_rnn_states_actor_ch = []
        values_ch = []
        nxt_rnn_states_critic_ch = []
        inx_data = 0
        for i in range(self.policy_num):
            if self.num_count[i]>0:
                indices_data = range(inx_data, inx_data + group_size*self.num_count[i])
                actions_temp, action_log_probs_temp, rnn_states_actor_temp = self.actor(obs_ch[indices_data],
                                                                        rnn_states_actor_ch[indices_data],
                                                                        masks_ch[indices_data],
                                                                        i,
                                                                        available_actions_ch if available_actions_ch is None else available_actions_ch[indices_data],
                                                                        deterministic)
                actions_ch.append(actions_temp)
                action_log_probs_ch.append(action_log_probs_temp)
                nxt_rnn_states_actor_ch.append(rnn_states_actor_temp)

                values_temp, rnn_states_critic_temp = self.critic(cent_obs_ch[indices_data], rnn_states_critic_ch[indices_data], masks_ch[indices_data], i)
                values_ch.append(values_temp)
                nxt_rnn_states_critic_ch.append(rnn_states_critic_temp)

            inx_data +=group_size * self.num_count[i]

        action_out = torch.cat(actions_ch, dim=0)
        action_log_probs_out = torch.cat(action_log_probs_ch, dim=0)
        rnn_states_actor_out = torch.cat(nxt_rnn_states_actor_ch, dim=0)
        values_out = torch.cat(values_ch, dim=0)
        rnn_states_critic_out = torch.cat(nxt_rnn_states_critic_ch, dim=0)

        values = values_out[inver_sort]
        actions = action_out[inver_sort]
        action_log_probs = action_log_probs_out[inver_sort]
        rnn_states_actor = rnn_states_actor_out[inver_sort]
        rnn_states_critic = rnn_states_critic_out[inver_sort]

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        if self.use_mixer:
            cent_obs = self.mixer(cent_obs)

        group_size = len(cent_obs) // self.num_indices
        ex_sort, inver_sort = self.expand_sort_line(group_size)
        cent_obs_ch = cent_obs[ex_sort]
        masks_ch = masks[ex_sort]
        rnn_states_critic_ch = rnn_states_critic[ex_sort]
        values_ch = []
        inx_data = 0

        for i in range(self.policy_num):
            if self.num_count[i]>0:
                indices_data = range(inx_data, inx_data + group_size*self.num_count[i])
                values_temp, _ = self.critic(cent_obs_ch[indices_data], rnn_states_critic_ch[indices_data], masks_ch[indices_data], i)
                values_ch.append(values_temp)

            inx_data +=group_size * self.num_count[i]

        values_out = torch.cat(values_ch, dim=0)
        values = values_out[inver_sort]

        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        if self.use_mixer:
            obs = self.mixer(obs)
            cent_obs = self.mixer(cent_obs)

        group_size = len(obs) // self.num_indices
        ex_sort, inver_sort = self.expand_sort_line(group_size)
        obs_ch = obs[ex_sort]
        cent_obs_ch = cent_obs[ex_sort]
        masks_ch = masks[ex_sort]
        rnn_states_actor_ch = rnn_states_actor[ex_sort]
        rnn_states_critic_ch = rnn_states_critic[ex_sort]
        action_ch = action[ex_sort]
        if available_actions is not None:
            available_actions_ch = available_actions[ex_sort]
        else:
            available_actions_ch = None
        if active_masks is not None:
            active_masks_ch = active_masks[ex_sort]
        else:
            active_masks_ch = None

        action_log_probs_ch = []
        dist_entropy_ch = []
        values_ch = []
        inx_data = 0

        for i in range(self.policy_num):
            if self.num_count[i]>0:
                indices_data = range(inx_data, inx_data + group_size*self.num_count[i])
                action_log_probs_temp, dist_entropy_temp = self.actor.evaluate_actions(obs_ch[indices_data],
                                                                                rnn_states_actor_ch[indices_data],
                                                                                action_ch[indices_data],
                                                                                masks_ch[indices_data],
                                                                                i,
                                                                                available_actions_ch if available_actions_ch is None else available_actions_ch[indices_data],
                                                                                active_masks_ch if active_masks_ch is None else active_masks_ch[indices_data])

                values_temp, _ = self.critic(cent_obs_ch[indices_data], rnn_states_critic_ch[indices_data], masks_ch[indices_data], i)
                values_ch.append(values_temp)
                action_log_probs_ch.append(action_log_probs_temp)
                dist_entropy_ch.append(dist_entropy_temp)

            inx_data +=group_size * self.num_count[i]

        values_out = torch.cat(values_ch, dim=0)
        action_log_probs_out = torch.cat(action_log_probs_ch, dim=0)
        dist_entropy_out = torch.cat(dist_entropy_ch, dim=0)

        values = values_out[inver_sort]
        action_log_probs = action_log_probs_out[inver_sort]
        dist_entropy = dist_entropy_out[inver_sort]

        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        if self.use_mixer:
            obs = self.mixer(obs)

        group_size = len(obs) // self.num_indices
        ex_sort, inver_sort = self.expand_sort_line(group_size)
        obs_ch = obs[ex_sort]
        masks_ch = masks[ex_sort]
        rnn_states_actor_ch = rnn_states_actor[ex_sort]

        if available_actions is not None:
            available_actions_ch = available_actions[ex_sort]
        else:
            available_actions_ch = None

        actions_ch = []
        nxt_rnn_states_actor_ch = []
        inx_data = 0

        for i in range(self.policy_num):
            if self.num_count[i]>0:
                indices_data = range(inx_data, inx_data + group_size*self.num_count[i])
                actions_temp, _, rnn_states_actor_temp = self.actor(obs_ch[indices_data], rnn_states_actor_ch[indices_data], masks_ch[indices_data], i, 
                                                                    available_actions_ch if available_actions_ch is None else available_actions_ch[indices_data], 
                                                                    deterministic)
                actions_ch.append(actions_temp)
                nxt_rnn_states_actor_ch.append(rnn_states_actor_temp)
            
            inx_data +=group_size * self.num_count[i]

        actions_out = torch.cat(actions_ch, dim = 0)
        rnn_states_actor_out  = torch.cat(nxt_rnn_states_actor_ch, dim = 0)
        actions = actions_out[inver_sort]
        rnn_states_actor = rnn_states_actor_out[inver_sort]

        return actions, rnn_states_actor
