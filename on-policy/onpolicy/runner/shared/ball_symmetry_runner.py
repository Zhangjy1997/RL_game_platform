import time
import numpy as np
import torch
import copy
from onpolicy.runner.shared.base_runner import Runner
from onpolicy.utils.memory_check import check_memory_usage
from onpolicy.envs.uav.scenarios.N_v_interaction import dict2vector
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy_sigma import R_MAPPOPolicy as Policy_sigma
from onpolicy.algorithms.r_mappo.r_mappo_sigma import R_MAPPO as TrainAlgo
from onpolicy.utils.shared_buffer import SharedReplayBuffer
from gym import spaces
import wandb
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

class BALL_Sigma_Runner(Runner):
    """Runner class to perform training, evaluation. and data collection for the UAVs. See parent class for details."""
    def __init__(self, config):
        super(BALL_Sigma_Runner, self).__init__(config)
        self.channel_interval = self.all_args.channel_interval
        if self.all_args.use_share_policy:
            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

            # policy network
            self.policy = Policy_sigma(self.all_args,
                               self.envs.observation_space[0],
                                share_observation_space,
                                self.envs.action_space[0],
                                device = self.device)

            # algorithm
            self.trainer = TrainAlgo(self.all_args, self.policy, device = self.device)

            shape_obs = self.envs.observation_space[0].shape
            shape_obs = shape_obs[-1]
            obs_fusion = spaces.Box(low=-1.0, high=1.0, shape=((shape_obs + self.all_args.population_size),), dtype=np.float32)
            shape_cent_obs = share_observation_space.shape
            shape_cent_obs = shape_cent_obs[-1]
            cent_obs_fusion = spaces.Box(low=-1.0, high=1.0, shape=(shape_cent_obs + self.all_args.population_size,), dtype=np.float32)

            self.buffer = SharedReplayBuffer(self.all_args,
                                            self.num_agents,
                                            obs_fusion,
                                            cent_obs_fusion,
                                            self.envs.action_space[0])
        
        
    def run(self):
        self.warmup()   
        self.trainer.policy.set_fusion_true()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        info_logs = dict()
        reward_logs = dict()
        num_logs = dict()
        self.reward_time_series = dict()
        
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # Divide environment information into separate groups based on policies.
                policy_head_str = ["policy_" + str(self.porbs_inx[i])+"_" for i in range(self.n_rollout_threads)]
                policy_infos = []
                for prefix, info in zip(policy_head_str, infos):
                    policy_info = {prefix + k: v for k, v in info.items()}
                    policy_infos.append(policy_info)
                for info in policy_infos:
                    for k in info:
                        info_logs[k] = info[k] if k not in info_logs else info[k] + info_logs[k]

                # Track rewards and training length (count) for each policy
                unique_indices, counts = np.unique(self.porbs_inx, return_counts=True)
                index_counts = dict(zip(unique_indices, counts))
                average_rewards = np.array([np.mean(arr) for arr in rewards])
                reward_sums = {index: average_rewards[self.porbs_inx == index].sum() for index in unique_indices}
                for k in reward_sums.keys():
                    reward_logs[k] = reward_sums[k] if k not in reward_logs else reward_sums[k] + reward_logs[k]
                    num_logs[k] = index_counts[k] if k not in num_logs else index_counts[k] + num_logs[k]

                # Re-sample training policies after the game ends
                if self.all_args.use_mix_policy:
                    all_done = np.all(dones, axis=1)
                    done_indices = np.where(all_done)[0]
                    # if step< self.episode_length - 1:
                    # For each done agent, randomly select a new policy
                    for i in done_indices:
                        random_inx = np.random.choice(self.id_sigma.shape[0])
                        self.porbs_mat[i] = self.id_sigma[random_inx]
                        self.porbs_inx[i] = self.id_inx[random_inx]
                        self.flatten_probs_mat[i] = self.flatten_id[random_inx]

                    # Implement training policy modifications
                    # self.envs.world.oppo_policy.update_index_multi_channels(done_indices)
                    self.envs.world.oppo_policy.set_probs_multi_channel(self.flatten_probs_mat, done_indices)
                
                expanded_probs = np.repeat(self.porbs_mat[:, np.newaxis, :], obs.shape[1], axis=1)
                obs = np.concatenate((obs, expanded_probs), axis=-1)
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)


            
            # compute return and update network
            self.compute()
            train_infos = self.train()

            # # Re-sample training policies
            # if self.all_args.use_mix_policy:
            #     for i in done_ends:
            #         random_inx = np.random.choice(self.id_sigma.shape[0])
            #         self.porbs_mat[i] = self.id_sigma[random_inx]
            #         self.porbs_inx[i] = self.id_inx[random_inx]
            #     self.envs.world.oppo_policy.set_probs_multi_channel(self.porbs_mat, done_indices)
            #     self.set_policy_sigma(self.porbs_mat)

            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if "BALL" in self.env_name:
                    env_infos = {}
                    # for agent_id in range(self.num_agents):
                    #     idv_rews = []
                    #     for info in infos:
                    #         if 'individual_reward' in info[agent_id].keys():
                    #             idv_rews.append(info[agent_id]['individual_reward'])
                    #     agent_k = 'agent%i/individual_rewards' % agent_id
                    #     env_infos[agent_k] = idv_rews

                ## update env reward infos
                match_str_head = ["policy_" + str(i)+"_" for i in self.id_inx]
                for k_head, inx in zip(match_str_head,self.id_inx):
                    for k in info_logs.keys():
                        if k_head in k:
                            train_infos[k] = info_logs[k] / num_logs[inx] * self.episode_length
                    if inx in reward_logs.keys():
                        avg_reward = reward_logs[inx] / num_logs[inx] * self.episode_length
                        train_infos[k_head+"average_episode_rewards"] = avg_reward
                        if inx not in self.reward_time_series:
                            self.reward_time_series[inx] = []
                        self.reward_time_series[inx].append(avg_reward)
                info_logs = dict()
                reward_logs = dict()
                num_logs = dict()
                # train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                # print("average episode rewards is {}".format(train_infos["policy_" + self.envs.world.team_name + str(self.policy_num)+"_"+"average_episode_rewards"]))
                self.log_train(train_infos, self.all_args.global_steps + total_num_steps)
                self.log_env(env_infos, self.all_args.global_steps + total_num_steps)
                

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    # Deprecated
    def get_eval_reward(self, N):
        eval_rewards = dict()
        for strategy, rewards in self.reward_time_series.items():
            if len(rewards) < N:
                N_b = len(rewards)
                eval_reward = sum(rewards) / N_b
            else:
                eval_reward = sum(rewards[-N:]) / N
            eval_rewards[strategy] = eval_reward
        return eval_rewards

    # Deprecated
    def check_training_effectiveness(self, N, threshold):
        all_effective = True
        self.strategy_effectiveness = {}

        for strategy, rewards in self.reward_time_series.items():
            if len(rewards) < 2 * N:
                self.strategy_effectiveness[strategy] = False
                all_effective = False
                continue

            avg_first_N = sum(rewards[:N]) / N
            avg_last_N = sum(rewards[-N:]) / N

            difference = avg_last_N - avg_first_N

            is_effective = difference > threshold
            self.strategy_effectiveness[strategy] = is_effective

            if not is_effective:
                all_effective = False

        return all_effective

    def set_policy_sigma(self, sigma):
        self.trainer.set_policy_sigma(sigma)

    # Set the policy id
    def set_id_sigma(self, id_sigma, id_inx = None, flatten_id_sigma = None):
        self.id_sigma = id_sigma
        if id_inx is None:
            self.id_inx = np.arange(1,len(self.id_sigma)+1,1)
        else:
            self.id_inx = id_inx
        if flatten_id_sigma is None:
            self.flatten_id = copy.deepcopy(id_sigma)
        else:
            self.flatten_id = flatten_id_sigma

        self.porbs_mat = np.zeros((self.n_rollout_threads,id_sigma.shape[-1]))
        self.flatten_probs_mat = np.zeros((self.n_rollout_threads, self.flatten_id.shape[-1]))
        self.porbs_inx = np.zeros(self.n_rollout_threads, dtype=int)
        for i in range(self.n_rollout_threads):
            random_inx = np.random.choice(id_sigma.shape[0])
            self.porbs_mat[i] = id_sigma[random_inx]
            self.flatten_probs_mat[i] = self.flatten_id[random_inx]
            self.porbs_inx[i] = self.id_inx[random_inx]
        self.envs.world.oppo_policy.set_probs_mat(self.flatten_probs_mat)
        self.set_policy_sigma(self.porbs_mat)

    def warmup(self):
        # reset env
        obs = self.envs.reset()
        expanded_probs = np.repeat(self.porbs_mat[:, np.newaxis, :], obs.shape[1], axis=1)
        obs = np.concatenate((obs, expanded_probs), axis=-1)
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

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                            np.concatenate(self.buffer.obs[step]),
                            np.concatenate(self.buffer.rnn_states[step]),
                            np.concatenate(self.buffer.rnn_states_critic[step]),
                            np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        actions_env = np.concatenate([actions[:, idx, :] for idx in range(self.num_agents)], axis=1)
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        # refer to smac and football environment!!!
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

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
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, bad_masks=bad_masks, active_masks=active_masks)
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

    # Calculates the victory outcome in a game between pursuers and evaders.
    def checkVictory(self, info):
        win = 0
        lose = 0
        draw = 0
        round_counter = 1
        win_list = ["PLAYER_0_win_reward", "PLAYER_1_lose_reward"]
        lose_list = ["PLAYER_0_lose_reward", "PLAYER_1_win_reward"]
        draw_list = ["PLAYER_0_draw", "PLAYER_1_draw"]

        # Assumes a single evader in the env
        for k in info:
            if any(sub in k for sub in win_list):
                win += 1
                # print(k)
                break
            if any(sub in k for sub in lose_list):
                lose += 1
                # print(k)
                break
            if any(sub in k for sub in draw_list):
                draw += 1
                # print(k)
                break
        draw += round_counter - (win + lose + draw)
        return win, lose, draw

    @torch.no_grad()
    def calu_win_prob(self, total_episodes):
        eval_obs = self.envs.reset()
        self.total_pursuer_win = 0
        self.total_evader_win = 0
        self.total_draw = 0
        self.total_round = 0
        self.total_N_array = np.zeros(total_episodes)
        self.total_reward = 0
        self.eva_r_list = []
        self.trainer.policy.set_fusion_false()
        eval_rnn_states = np.zeros((self.n_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for episodes in range(total_episodes):
            for eval_step in range(self.episode_length):
                self.trainer.prep_rollout()
                eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                    np.concatenate(eval_rnn_states),
                                                    np.concatenate(eval_masks),
                                                    deterministic=True)
                eval_actions = np.array(np.split(_t2n(eval_action), self.n_rollout_threads))
                eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_rollout_threads))
                eval_actions_env = np.concatenate([eval_actions[:, idx, :] for idx in range(self.num_agents)], axis=1)

                # Obser reward and next obs
                # print("action network:", eval_actions_env)
                eval_obs, eval_rewards, eval_dones, eval_infos = self.envs.step(eval_actions_env)
                # print(eval_infos)

                for i in range(self.n_rollout_threads):
                    self.total_reward += eval_rewards[i][0][0]
                    if eval_dones[i].all():
                        p_win, e_win, draw = self.checkVictory(eval_infos[i])
                        self.total_pursuer_win += p_win
                        self.total_evader_win += e_win
                        self.total_draw += draw
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
    def get_payoff_sigma(self, total_episodes, delta = None, low_i = -1):
        eval_payoffs = dict()
        standard_vaules = dict()
        for idx, row, probs_row in zip(self.id_inx, self.id_sigma, self.flatten_id):
            if idx < low_i:
                continue
            self.set_policy_sigma(np.tile(row,(1,1)))
            self.envs.world.oppo_policy.set_probs_all(probs_row)
            print("eval_policy {}:".format(idx))
            if delta is None:
                delta = np.inf

            total_reward_ = 0
            total_round_ = 0
            eval_r_list_ = []
            
            while True:
                self.calu_win_prob(total_episodes)
                total_reward_ += self.total_reward
                total_round_ += self.total_round
                eval_r_list_ += copy.deepcopy(self.eva_r_list)
                payoff_p = (total_reward_)/(total_round_)
                std_ = np.std(np.array(eval_r_list_))/np.sqrt(len(eval_r_list_))
                print("standard value = {}, target = {}".format(std_, delta))
                if std_ < delta:
                    break
        
            eval_payoffs[idx] = payoff_p
            standard_vaules[idx] = std_
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
