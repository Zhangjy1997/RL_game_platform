import numpy as np
import torch
import wandb
from onpolicy.algorithms.NeuPL.eval_match import eval_match_uav as eval_match
from onpolicy.algorithms.NeuPL.Policy_prob_matrix import Nash_matrix as prob_matrix
from onpolicy.algorithms.NeuPL.mixing_policy import Parallel_mixing_policy as mixing_policy
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

def restore_eval_policy(policy, model_dir, label_str, use_mixer = True):
    """Restore policy's networks from a saved model."""
    policy_actor_state_dict = torch.load(str(model_dir) + '/actor_' + label_str + '.pt')
    policy.actor.load_state_dict(policy_actor_state_dict)
    if use_mixer:
        policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer_' + label_str + '.pt')
        policy.mixer.load_state_dict(policy_mixer_state_dict)

def creat_policy(args, envs, save_dir, label_str, device, use_mixer):
    policy = Policy(args,
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_act_space[0],
                        device)
    restore_eval_policy(policy, save_dir, label_str, use_mixer)
    return policy

class Neural_population_learning:
    def __init__(self, args, policies_p1, policies_p2, runners, eval_envs, role_names, save_dir):
        self.args = args
        self.policies_p1 = policies_p1
        self.policies_p2 = policies_p2
        self.runners = runners
        self.eval_envs = eval_envs
        self.role_names = role_names
        self.save_dir = save_dir
        self.n_threads = self.args.n_rollout_threads
        self.n_eval_eps = self.args.eval_episode_num
        self.g_step = 0
        self.total_round = self.args.total_round
        self.use_wandb = self.args.use_wandb
        self.num_env_steps = self.args.num_env_steps
        self.eval = eval_match(self.policies_p1, self.policies_p2, self.eval_envs)
        self.p1_num = len(policies_p1)
        self.p2_num = len(policies_p2)
        assert self.p1_num == self.p2_num, "dimension error!"
        self.p1_space = [role_names[0] + str(i) for i in range(self.p1_num)]
        self.p2_space = [role_names[1] + str(i) for i in range(self.p2_num)]
        self.graph_generator = prob_matrix(self.p1_space, self.p2_space)

    def run(self, use_warmup = True, train_inx = None, use_inherit = True):
        if use_warmup:
            self.warmup()
        print("Training start!")
        for i in range(self.total_round):
            if i >= 1:
                use_inherit = True
            # refresh policies
            for j in range(1, self.p1_num):
                restore_eval_policy(self.policies_p1[j], self.save_dir, self.p1_space[j], self.policies_p1[j].use_mixer)
                restore_eval_policy(self.policies_p2[j], self.save_dir, self.p2_space[j], self.policies_p2[j].use_mixer)

            dict_mat_p1 = dict()
            dict_mat_p2 = dict()
            self.eval = eval_match(self.policies_p1, self.policies_p2, self.eval_envs)
            self.eval.get_win_prob_mat(self.n_threads, self.n_eval_eps)
            # payoff_mat = np.log((self.eval.win_prob_mat + 1e-10)/(1-self.eval.win_prob_mat + 1e-10))
            payoff_mat = self.eval.total_round_mat * (2 * self.eval.win_prob_mat -1)/(self.n_threads*self.n_eval_eps)
            self.graph_generator.update_prob_matrix_simple(payoff_mat)
            dict_mat_p1["win_prob_mat_" + str(i)] = self.eval.win_prob_mat
            dict_mat_p1["probs_p1_mat_" + str(i)] = self.graph_generator.p1_prob_mat
            dict_mat_p1["payoff_p1_mat_" + str(i)] = payoff_mat
            dict_mat_p2["probs_p2_mat_" + str(i)] = self.graph_generator.p2_prob_mat

            print("prob_p1=")
            print(self.graph_generator.p1_prob_mat)
            print("prob_p2=")
            print(self.graph_generator.p2_prob_mat)

            if self.use_wandb:
                for k, v in dict_mat_p1.items():
                    wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p2_space)})
                for k, v in dict_mat_p2.items():
                    wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p1_space)})
                wandb.log({"train_rounds": i})

            if train_inx is None:
                for j in range(1, self.p1_num):
                    self.step_run(j, use_inherit=use_inherit)
            else:
                for j in train_inx:
                    self.step_run(j, use_inherit=use_inherit)

            # for j in range(1, self.p1_num):
            #     oppo_policies = self.policies_p2[0: j]
            #     mix_policy = mixing_policy(self.n_threads, oppo_policies, self.graph_generator.p2_prob_mat[j, 0:j])
            #     self.runners[0].all_args.runner_num = j
            #     #self.runners[0].restore()
            #     self.runners[0].inherit_policy(self.save_dir, self.p1_space[j])
            #     self.runners[0].all_args.global_steps = self.g_step
            #     self.runners[0].envs.world.oppo_policy = mix_policy
            #     self.runners[0].run()
            #     self.g_step += self.num_env_steps

            #     oppo_policies = self.policies_p1[0: j]
            #     mix_policy = mixing_policy(self.n_threads, oppo_policies, self.graph_generator.p1_prob_mat[j, 0:j])
            #     self.runners[1].all_args.runner_num = j
            #     #self.runners[1].restore()
            #     self.runners[1].inherit_policy(self.save_dir, self.p2_space[j])
            #     self.runners[1].all_args.global_steps = self.g_step
            #     self.runners[1].envs.world.oppo_policy = mix_policy
            #     self.runners[1].run()
            #     self.g_step += self.num_env_steps
            
        self.eval_round_step(self.total_round)

    def run_single_round(self, train_inx = None, round_num = 1, use_inherit = True):
        if train_inx is None:
            train_inx = range(self.p1_num)

        self.eval_round_step(round_num=round_num)

        for i in train_inx:
            self.step_run(i, use_inherit=use_inherit)


    # TODO: 
    def step_run(self, inx, use_inherit = True):
        oppo_policies = self.policies_p2[0: inx]
        mix_policy = mixing_policy(self.n_threads, oppo_policies, self.graph_generator.p2_prob_mat[inx, 0:inx])
        self.runners[0].all_args.runner_num = inx
        #self.runners[0].restore()
        if use_inherit:
            self.runners[0].inherit_policy(self.save_dir, self.p1_space[inx])
        self.runners[0].all_args.global_steps = self.g_step
        self.runners[0].envs.world.oppo_policy = mix_policy
        self.runners[0].run()
        self.g_step += self.num_env_steps

        oppo_policies = self.policies_p1[0: inx]
        mix_policy = mixing_policy(self.n_threads, oppo_policies, self.graph_generator.p1_prob_mat[inx, 0:inx])
        self.runners[1].all_args.runner_num = inx
        #self.runners[1].restore()
        if use_inherit:
            self.runners[1].inherit_policy(self.save_dir, self.p2_space[inx])
        self.runners[1].all_args.global_steps = self.g_step
        self.runners[1].envs.world.oppo_policy = mix_policy
        self.runners[1].run()
        self.g_step += self.num_env_steps

        
    def eval_round_step(self, round_num):
        for j in range(1, self.p1_num):
                restore_eval_policy(self.policies_p1[j], self.save_dir, self.p1_space[j], self.policies_p1[j].use_mixer)
                restore_eval_policy(self.policies_p2[j], self.save_dir, self.p2_space[j], self.policies_p2[j].use_mixer)

        dict_mat_p1 = dict()
        dict_mat_p2 = dict()
        self.eval = eval_match(self.policies_p1, self.policies_p2, self.eval_envs)
        self.eval.get_win_prob_mat(self.n_threads, self.n_eval_eps)
        # payoff_mat = np.log((self.eval.win_prob_mat + 1e-10)/(1-self.eval.win_prob_mat + 1e-10))
        payoff_mat = self.eval.total_round_mat * (2 * self.eval.win_prob_mat -1)/(self.n_threads*self.n_eval_eps)
        self.graph_generator.update_prob_matrix_simple(payoff_mat)
        dict_mat_p1["win_prob_mat_" + str(round_num)] = self.eval.win_prob_mat
        dict_mat_p1["probs_p1_mat_" + str(round_num)] = self.graph_generator.p1_prob_mat
        dict_mat_p1["payoff_p1_mat_" + str(round_num)] = payoff_mat
        dict_mat_p2["probs_p2_mat_" + str(round_num)] = self.graph_generator.p2_prob_mat

        print("prob_p1=")
        print(self.graph_generator.p1_prob_mat)
        print("prob_p2=")
        print(self.graph_generator.p2_prob_mat)

        if self.use_wandb:
            for k, v in dict_mat_p1.items():
                wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p2_space)})
            for k, v in dict_mat_p2.items():
                wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p1_space)})
            wandb.log({"train_rounds": round_num})


    
    def warmup(self, start_inx = 1):
        print("NeuPL warming up!")
        for i in range(start_inx, self.p1_num):
            dict_mat_p1 = dict()
            dict_mat_p2 = dict()
            self.eval = eval_match(self.policies_p1, self.policies_p2, self.eval_envs)
            sub_array_1 = [0, i, 0, i]
            # sub_array_2 = [i - 1, i, 0, i]
            print("eval start!")
            self.eval.get_win_prob_mat(self.n_threads, self.n_eval_eps, sub_array_1)
            #self.eval.get_win_prob_mat(self.n_threads, self.n_eval_eps, sub_array_2)
            # payoff_mat = np.log((self.eval.win_prob_mat + 1e-10)/(1-self.eval.win_prob_mat + 1e-10))
            payoff_mat = self.eval.total_round_mat * (2 * self.eval.win_prob_mat -1)/(self.n_threads*self.n_eval_eps)
            self.graph_generator.update_prob_matrix_simple(payoff_mat)
            dict_mat_p1["warm_up_win_prob_mat_" + str(i)] = self.eval.win_prob_mat
            dict_mat_p1["warm_up_probs_p1_mat_" + str(i)] = self.graph_generator.p1_prob_mat
            dict_mat_p1["warm_up_payoff_p1_mat_" + str(i)] = payoff_mat
            dict_mat_p2["warm_up_probs_p2_mat_" + str(i)] = self.graph_generator.p2_prob_mat

            print("prob_p1=")
            print(self.graph_generator.p1_prob_mat)
            print("prob_p2=")
            print(self.graph_generator.p2_prob_mat)
            
            if self.use_wandb:
                for k, v in dict_mat_p1.items():
                    wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p2_space)})
                for k, v in dict_mat_p2.items():
                    wandb.log({k: wandb.Table(data=v.tolist(), columns = self.p1_space)})
                wandb.log({"warm_up_rounds": i})

            # train player1's policy
            oppo_policies = self.policies_p2[0: i]
            mix_policy = mixing_policy(self.n_threads, oppo_policies, self.graph_generator.p2_prob_mat[i, 0:i])
            self.runners[0].all_args.runner_num = i
            self.runners[0].all_args.global_steps = self.g_step
            self.runners[0].envs.world.oppo_policy = mix_policy
            self.runners[0].run()
            self.g_step += self.num_env_steps

            restore_eval_policy(self.policies_p1[i], self.save_dir, self.p1_space[i], self.policies_p1[i].use_mixer)

            # train player2's policy
            oppo_policies = self.policies_p1[0: i]
            mix_policy = mixing_policy(self.n_threads, oppo_policies, self.graph_generator.p1_prob_mat[i, 0:i])
            self.runners[1].all_args.runner_num = i
            self.runners[1].all_args.global_steps = self.g_step
            self.runners[1].envs.world.oppo_policy = mix_policy
            self.runners[1].run()
            self.g_step += self.num_env_steps

            restore_eval_policy(self.policies_p2[i], self.save_dir, self.p2_space[i], self.policies_p2[i].use_mixer)

