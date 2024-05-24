import numpy
import torch
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Oppo_Policy

def restore_eval_policy(oppo_policy, model_dir, label_str, use_mixer = True):
    """Restore policy's networks from a saved model."""
    policy_actor_state_dict = torch.load(str(model_dir) + '/actor_' + label_str + '.pt')
    oppo_policy.actor.load_state_dict(policy_actor_state_dict)
    if use_mixer:
        policy_mixer_state_dict = torch.load(str(model_dir) + '/mixer_' + label_str + '.pt')
        oppo_policy.mixer.load_state_dict(policy_mixer_state_dict)

def creat_oppo_policy(args, envs, save_dir, label_str, device):
    oppo_policy = Oppo_Policy(args,
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_obs_space[0],
                        envs.world.oppo_act_space[0],
                        device)
    restore_eval_policy(oppo_policy, save_dir, label_str, args.use_mixer)
    return oppo_policy


class train_alternate:
    def __init__(self, args, num_policy, runner_p, runner_e, save_dir, device):
        self.num_policy = num_policy
        self.all_args = args

        self.runner_p = runner_p
        self.runner_e = runner_e
        self.save_dir = save_dir
        
        self.name_p = self.runner_p.envs.world.team_name
        self.name_e = self.runner_e.envs.world.team_name
        self.device = device

    def run(self):
        for i in range(self.num_policy):
            self.all_args.num_agents = self.runner_p.envs.world.num_oppo
            self.all_args.use_mixer = True if self.all_args.num_agents >=2 else False
            if i != 0:
                self.runner_p.envs.world.oppo_policy = creat_oppo_policy(self.all_args, self.runner_p.envs, self.save_dir, str(self.name_e) + str(self.all_args.runner_num - 1), self.device)

            self.runner_p.all_args = self.all_args
            self.runner_p.run()

            self.all_args.global_steps += self.all_args.num_env_steps
            

            self.all_args.num_agents = self.runner_e.envs.world.num_oppo
            self.all_args.use_mixer = True if self.all_args.num_agents >=2 else False
            self.runner_e.envs.world.oppo_policy = creat_oppo_policy(self.all_args, self.runner_e.envs, self.save_dir, str(self.name_p) + str(self.all_args.runner_num), self.device)

            self.runner_e.all_args = self.all_args
            self.runner_e.run()

            self.all_args.global_steps += self.all_args.num_env_steps
            self.all_args.runner_num +=1
            
