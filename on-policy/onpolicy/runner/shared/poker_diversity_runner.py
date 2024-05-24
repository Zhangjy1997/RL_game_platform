import time
import numpy as np
import torch
import copy
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.utils.memory_check import check_memory_usage
from onpolicy.envs.uav.scenarios.N_v_interaction import dict2vector
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_diversity import R_MAPPOPolicy as Policy
from onpolicy.algorithms.r_mappo.r_mappo_diversity import R_MAPPO as TrainAlgo
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from gym import spaces
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class POKER_Runner(Runner):
    """Runner class to perform training, evaluation. and data collection for the UAVs. See parent class for details."""
    def __init__(self, config):
        super(POKER_Runner, self).__init__(config)
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,
                            self.envs.observation_space[0],
                            share_observation_space,
                            self.envs.action_space[0],
                            device = self.device)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

        self.kl_div_coef = self.all_args.kl_div_coef
        
        
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        info_logs = dict()
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, kl_divs = self.collect(step)
                # Obser reward and next obs
                obs, rewards, dones, infos, available_actions = self.envs.step(actions_env)
                data = obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic, kl_divs

                if self.all_args.use_mix_policy:
                    all_done = np.all(dones, axis=1)
                    done_indices = np.where(all_done)[0]
                    self.envs.world.oppo_policy.update_index_multi_channels(done_indices)

                # if self.all_args.use_mix_policy:
                #     for i in range(self.all_args.n_rollout_threads):
                #         if dones[i].all():
                #             self.envs.world.oppo_policy.updata_index_channel(i)

                # insert data into buffer
                self.insert(data)
                for info in infos:
                    for k in info:
                        info_logs[k] = info[k] if k not in info_logs else info[k] + info_logs[k]
            
            kl_fl = self.buffer.kl_divs.reshape(*self.buffer.kl_divs.shape[:2], -1)
            kl_sum = np.sum(np.sum(kl_fl, axis = -1), axis = 0)
            min_policy_inx = np.argmin(kl_sum)
            # if np.any(np.isnan(self.buffer.kl_divs)) or np.any(np.isinf(self.buffer.kl_divs)):
            #     print("nan or inf")
            # else:
            #     print("no nan and inf")
            self.buffer.rewards += self.kl_div_coef * self.buffer.kl_divs[:, min_policy_inx]
            self.trainer.update_anchor_policy([self.oppo_policies[min_policy_inx]])
            
            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, policy {}, num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                self.policy_inx,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if "POKER" in self.env_name:
                    env_infos = {}
                    # for agent_id in range(self.num_agents):
                    #     idv_rews = []
                    #     for info in infos:
                    #         if 'individual_reward' in info[agent_id].keys():
                    #             idv_rews.append(info[agent_id]['individual_reward'])
                    #     agent_k = 'agent%i/individual_rewards' % agent_id
                    #     env_infos[agent_k] = idv_rews

                ## update env reward infos
                policy_head = "policy_" + str(self.policy_inx) + "_"
                for k in info_logs.keys():
                    train_infos[policy_head + k] = info_logs[k] / self.n_rollout_threads
                info_logs = dict()
                train_infos[policy_head + "average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos[policy_head + "average_episode_rewards"]))
                self.log_train(train_infos, self.all_args.global_steps + total_num_steps)
                self.log_env(env_infos, self.all_args.global_steps + total_num_steps)
                

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def set_policy_inx(self, inx):
        self.policy_inx = inx

    def set_oppo_policies(self, oppo_policies):
        self.oppo_policies = oppo_policies
        self.buffer.kl_divs = np.zeros(
            (self.episode_length, len(oppo_policies), self.n_rollout_threads, 1, 1), dtype=np.float32)

    def warmup(self):
        # reset env
        obs, available_actions = self.envs.reset()
        # replay buffer
        if self.use_centralized_V:
            # TODO: implement real shared obs
            # share_obs = obs.reshape(self.n_rollout_threads, -1)
            # share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            # print("shared obs", share_obs.shape)
            share_obs = obs
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]),
                            np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        kl_div = self.calc_kl_div(step, actions)
        kl_divs = np.array(_t2n(kl_div.unsqueeze(-1)))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        actions_env = np.concatenate([actions[:, idx, :] for idx in range(self.num_agents)], axis=1)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, kl_divs
    
    def calc_kl_div(self, step, actions):
        
        kl_divs = self.trainer.policy.get_kl_divergence(np.concatenate(self.buffer.obs[step]),
                                                        np.concatenate(self.buffer.rnn_states[step]),
                                                        np.concatenate(actions),
                                                        np.concatenate(self.buffer.masks[step]),
                                                        self.oppo_policies,
                                                        np.concatenate(self.buffer.available_actions[step]),
                                                        np.concatenate(self.buffer.active_masks[step]))
        
        return kl_divs

    def insert(self, data):
        # refer to smac and football environment!!!
        obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, rnn_states, rnn_states_critic, kl_divs = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # TODO: compute activate mask according to infos
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # TODO: compute bad_mask according to infos
        bad_masks = np.array([[[0.0] if "TimeLimit.truncated" in info else [1.0] for _ in range(self.num_agents)] for info in infos])

        if self.use_centralized_V:
            # TODO: implement real shared obs
            # share_obs = obs.reshape(self.n_rollout_threads, -1)
            # share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
            share_obs = obs
        else:
            share_obs = obs
        self.buffer.kl_divs[self.buffer.step] = kl_divs.copy()
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, bad_masks=bad_masks, active_masks=active_masks, available_actions=available_actions)
        # self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, bad_masks=bad_masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        self.oppo_record= []
        self.team_record= []
        self.other_record = []
        eval_obs = self.eval_envs.reset()

        #self.position_record.append([eval_obs[0][0][0:3],eval_obs[0][1][0:3], eval_obs[0][2][0:3]])
        ob_op=copy.deepcopy(self.eval_envs.oppo_obs)
        ob_other = copy.deepcopy(self.eval_envs.other_obs)
        self.oppo_record.append([ob_op[0][i][0:3] for i in range(1)])
        self.team_record.append([eval_obs[0][i][0:3] for i in range(1)])
        self.other_record.append([ob_other[0][i][0:3] for i in range(1)])

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                np.concatenate(eval_rnn_states),
                                                np.concatenate(eval_masks),
                                                deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))
            eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(self.num_agents)], axis=1)

            # Obser reward and next obs
            # print("action network:", eval_actions_env)
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            # print(eval_infos)
            if eval_dones[0].all():
                #exit(0)
                pursuer_win, evader_win = self.checkVictory(eval_infos[0])
                if pursuer_win > evader_win:
                    print("pursuer win!")
                else:
                    print("evader win!")
                break

            eval_episode_rewards.append(eval_rewards)

            #record trajectory of pursuers and evaders 
            ob_op=copy.deepcopy(self.eval_envs.oppo_obs)
            ob_other = copy.deepcopy(self.eval_envs.other_obs)
            self.oppo_record.append([ob_op[0][i][0:3] for i in range(1)])
            self.team_record.append([eval_obs[0][i][0:3] for i in range(1)])
            self.other_record.append([ob_other[0][i][0:3] for i in range(1)])

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards'] = np.sum(np.array(eval_episode_rewards), axis=0)
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)


    @torch.no_grad()
    def calu_win_prob(self, total_episodes):
        eval_obs, eval_a_actions = self.envs.reset()
        self.total_round = 0
        self.total_N_array = np.zeros(total_episodes)
        self.total_reward = 0
        self.eva_r_list = []
        eval_rnn_states = np.zeros((self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for episodes in range(total_episodes):
            for eval_step in range(self.episode_length):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    np.concatenate(eval_a_actions),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(self.num_agents)], axis=1)

                # Obser reward and next obs
                # print("action network:", eval_actions_env)
                eval_obs, eval_rewards, eval_dones, eval_infos, eval_a_actions = self.envs.step(eval_actions_env)
                # print(eval_infos)

                for i in range(self.n_rollout_threads):
                    self.total_reward += eval_rewards[i][0][0]
                    if eval_dones[i].all():
                        self.total_round += 1
                        self.total_N_array[episodes] += 1
                        self.eva_r_list.append(eval_rewards[i][0][0])
                        if self.all_args.use_mix_policy: 
                            self.envs.world.oppo_policy.update_index_channel(i)

                eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            print("episodes: {}/{}".format(episodes, total_episodes))

    # eval the payoff for each policy
    def get_payoff_sigma(self, total_episodes):
        eval_payoffs = 0
        standard_vaules = 0

        print("eval_policy {}:".format(self.policy_inx))
        self.calu_win_prob(total_episodes)
        payoff_p = (self.total_reward)/(self.total_round)
        eval_payoffs = payoff_p
        standard_vaules = np.std(np.array(self.eva_r_list))/np.sqrt(len(self.eva_r_list))
        # Var_N = np.var(self.total_N_array, ddof=1) * total_episodes
        # standard_vaules = np.sqrt(4*self.total_round*max(prob_bar*(1-prob_bar),0.05) + Var_N * (2*prob_bar -1) ** 2)/(total_episodes * self.n_rollout_threads)
        return eval_payoffs, standard_vaules


    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir)  + "/actor"  + ".pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic" + ".pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm" + ".pt")
        if self.trainer.policy.use_mixer:
            policy_mixer = self.trainer.policy.mixer
            torch.save(policy_mixer.state_dict(), str(self.save_dir) + "/mixer" + ".pt")

    def save_as_filename(self, head_str):
        """Save policy's actor and critic networks."""
        label_str = head_str
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir)  + "/actor_" +  label_str + ".pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_" + label_str + ".pt")
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm_" + label_str + ".pt")
        if self.trainer.policy.use_mixer:
            policy_mixer = self.trainer.policy.mixer
            torch.save(policy_mixer.state_dict(), str(self.save_dir) + "/mixer_" + label_str + ".pt")

    def inherit_policy(self, policy_str):
        policy_actor_state_dict = torch.load(policy_str + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if self.all_args.use_mixer:
            policy_mixer_state_dict = torch.load(policy_str + '/mixer.pt')
            self.policy.mixer.load_state_dict(policy_mixer_state_dict)

        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(policy_str + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            if self.trainer._use_valuenorm:
                policy_vnorm_state_dict = torch.load(policy_str + '/vnorm.pt')
                self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_str = str(self.model_dir)
        self.inherit_policy(policy_str)
    
    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        raise NotImplementedError
        # envs = self.envs
        
        # all_frames = []
        # for episode in range(self.all_args.render_episodes):
        #     obs = envs.reset()
        #     if self.all_args.save_gifs:
        #         image = envs.render('rgb_array')[0][0]
        #         all_frames.append(image)
        #     else:
        #         envs.render('human')

        #     rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        #     masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            
        #     episode_rewards = []
            
        #     for step in range(self.episode_length):
        #         calc_start = time.time()

        #         self.trainer.prep_rollout()
        #         action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
        #                                             np.concatenate(rnn_states),
        #                                             np.concatenate(masks),
        #                                             deterministic=True)
        #         actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        #         rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        #         actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]

        #         # Obser reward and next obs
        #         obs, rewards, dones, infos = envs.step(actions_env)
        #         episode_rewards.append(rewards)

        #         rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        #         masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        #         masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        #         if self.all_args.save_gifs:
        #             image = envs.render('rgb_array')[0][0]
        #             all_frames.append(image)
        #             calc_end = time.time()
        #             elapsed = calc_end - calc_start
        #             if elapsed < self.all_args.ifi:
        #                 time.sleep(self.all_args.ifi - elapsed)
        #         else:
        #             envs.render('human')

        #     print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
