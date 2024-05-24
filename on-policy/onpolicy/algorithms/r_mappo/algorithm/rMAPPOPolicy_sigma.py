import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic_sigma import R_Actor_sigma, R_Critic_sigma
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.order_moudle_sort import Order_Mixer
from onpolicy.algorithms.utils.util import check
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
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.use_mixer = args.use_mixer

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.sigma_tensor = None
        self.population_size = args.population_size
        self.sigma_fusion = True

        if self.use_mixer:
            self.team_name = args.team_name
            self.mixer=Order_Mixer(args, self.team_name, self.device)
            self.actor = R_Actor_sigma(args, self.mixer.output_space(), self.act_space, self.device)
            self.critic = R_Critic_sigma(args, self.mixer.output_space(), self.device)
        else:
            self.actor = R_Actor_sigma(args, self.obs_space, self.act_space, self.device)
            self.critic = R_Critic_sigma(args, self.share_obs_space, self.device)

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

    def set_sigma(self, sigma_tensor):
        sigma_tensor = check(sigma_tensor).to(**self.tpdv)
        self.sigma_tensor = sigma_tensor

    def set_fusion_true(self):
        self.sigma_fusion = True

    def set_fusion_false(self):
        self.sigma_fusion = False


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
        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.population_size], axis=-1)
            cent_obs, _ = np.split(cent_obs, [cent_obs.shape[-1] - self.population_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        if self.use_mixer:
            obs=self.mixer(obs)
            cent_obs=self.mixer(cent_obs)

        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                rnn_states_actor,
                                                                masks,
                                                                sigma,
                                                                available_actions,
                                                                deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, sigma)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        if self.sigma_fusion:
            cent_obs, sigma = np.split(cent_obs, [cent_obs.shape[-1] - self.population_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        if self.use_mixer:
            cent_obs = self.mixer(cent_obs)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks, sigma)
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
        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.population_size], axis=-1)
            cent_obs, _ = np.split(cent_obs, [cent_obs.shape[-1] - self.population_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        if self.use_mixer:
            obs = self.mixer(obs)
            cent_obs = self.mixer(cent_obs)

        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     sigma,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks, sigma)
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
        if self.sigma_fusion:
            obs, sigma = np.split(obs, [obs.shape[-1] - self.population_size], axis=-1)
        else:
            sigma = self.sigma_tensor

        if self.use_mixer:
            obs = self.mixer(obs)

        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, sigma, available_actions, deterministic)
        return actions, rnn_states_actor