import numpy as np
import torch
import torch.nn as nn
from onpolicy.algorithms.utils.mlp import MLPLayer
from onpolicy.algorithms.utils.attn_moudle import MultiHeadAttention
import copy
import argparse
from gym import spaces
from onpolicy.algorithms.utils.util import check

class Encoder(MLPLayer):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, use_feature_normalization):
        super(Encoder, self).__init__(input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU)
        self._use_feature_normalization=use_feature_normalization

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)
    
    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)
        x=super().forward(x)
        return x


class Order_Mixer(nn.Module):
    def __init__(self, args, role_name, device=torch.device("cpu")):
        super(Order_Mixer, self).__init__()
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        #self._stacked_frames = args.stacked_frames
        self._layer_N = args.encoder_layer_N
        self.hidden_size = args.encoder_hidden_size
        self.num_players=args.num_agents
        self.attn_size=args.attn_size
        self.proprio_shape=args.proprio_shape
        self.teammate_shape=args.teammate_shape
        self.opponent_shape=args.opponent_shape
        self.n_head=args.n_head
        self.role_name=role_name
        self.d_k=args.d_k
        self.tpdv = dict(dtype=torch.float32, device=device)

        # self.n_pair=(self.num_players-1)*self.num_players
        # init encoders
        self.proprio_encoder=Encoder(self.proprio_shape, self.hidden_size, self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)
        self.teammate_encoder=Encoder(self.teammate_shape,self.hidden_size,self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)
        self.opponent_encoder=Encoder(self.opponent_shape,self.hidden_size,self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)

        self.query_team_encoder=Encoder(self.hidden_size, self.attn_size, self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)
        self.query_oppo_encoder=Encoder(self.hidden_size, self.attn_size, self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)
        #self.query_encoder=Encoder(self.proprio_shape, self.attn_size, self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)
        self.value_team_encoder=Encoder(self.hidden_size, self.attn_size, self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)
        self.value_oppo_encoder=Encoder(self.hidden_size, self.attn_size, self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)
        self.key_team_encoder=Encoder(self.hidden_size, self.attn_size, self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)
        self.key_oppo_encoder=Encoder(self.hidden_size, self.attn_size, self._layer_N, self._use_orthogonal, self._use_ReLU, self._use_feature_normalization)

        self.MH_attn_team=MultiHeadAttention(self.n_head, self.attn_size, self.d_k, self.d_k)
        self.MH_attn_oppo=MultiHeadAttention(self.n_head, self.attn_size, self.d_k, self.d_k)

        if 'pursuer' in self.role_name:
            self.role_keys=['proprioceptive'] + ['teammate_'+str(i) for i in range(self.num_players-1)] + ['opponent_0']
            self.exist_team = True
            self.exist_oppo = True
            self.output_size = self.hidden_size + 2*self.attn_size
            self.num_team = self.num_players - 1
            self.num_oppo = 1
        elif 'evader' in self.role_name:
            self.role_keys=['proprioceptive'] + ['opponent_'+str(i) for i in range(self.num_players)]
            self.exist_team = False
            self.exist_oppo = True
            self.output_size = self.hidden_size + self.attn_size
            self.num_team = 0
            self.num_oppo = self.num_players
        else:
            print("wrong role name!")

        self.to(device)

    def forward(self, obs):
        obs = check(obs).to(**self.tpdv)
        obs_subdim_vals = [None for _ in range(self.num_players+1)]
        obs_subdim=dict(zip(self.role_keys, obs_subdim_vals))

        for k in self.role_keys:
            if 'proprio' in k:
                obs_subdim[k]=self.proprio_shape
            elif 'teammate' in k:
                obs_subdim[k]=self.teammate_shape
            elif 'opponent' in k:
                obs_subdim[k]=self.opponent_shape
            else:
                print("role keys error!")

        self.obs_subdim=copy.deepcopy(obs_subdim)
        obs_role_list=torch.split(obs,list(obs_subdim.values()),dim=-1)
        obs_role_dict=dict(zip(self.role_keys,obs_role_list))

        obs_code_list=[None for _ in range(self.num_players+1)]
        obs_code=dict(zip(self.role_keys,obs_code_list))

        for k in self.role_keys:
            if 'proprio' in k:
                obs_code[k] = self.proprio_encoder(obs_role_dict[k])
                #obs_code[k] = obs_role_dict[k]
            elif 'teammate' in k:
                obs_code[k] = self.teammate_encoder(obs_role_dict[k])
            elif 'opponent' in k:
                obs_code[k] = self.opponent_encoder(obs_role_dict[k])
            else:
                print("role keys error!")
        self.obs_single_code=obs_code
        self.attn_code()

        self.output_list = []
        self.output_list.append(obs_code['proprioceptive'])
        if self.exist_team:
            self.query_team_code=self.query_team_encoder(obs_code['proprioceptive'])
            self.query_team_code=self.query_team_code.unsqueeze(-2)
            obs_team, attn_m = self.MH_attn_team(self.query_team_code, self.key_team_codes, self.value_team_codes)
            obs_team=obs_team.squeeze(-2)
            self.output_list.append(obs_team)
        if self.exist_oppo:
            self.query_oppo_code=self.query_oppo_encoder(obs_code['proprioceptive'])
            self.query_oppo_code=self.query_oppo_code.unsqueeze(-2)
            obs_oppo, attn_m = self.MH_attn_oppo(self.query_oppo_code, self.key_oppo_codes, self.value_oppo_codes)
            obs_oppo=obs_oppo.squeeze(-2)
            self.output_list.append(obs_oppo)
        # obs_team=obs_team.squeeze(-2)
        obs_out = torch.cat(self.output_list, dim = -1)
        return obs_out
    
    def attn_code(self):
        if self.exist_team:
            team_code_list =[None for _ in range(self.num_team)]
            team_key_code_list=[None for _ in range(self.num_team)]
            team_num = 0
        if self.exist_oppo:
            oppo_code_list =[None for _ in range(self.num_oppo)]
            oppo_key_code_list=[None for _ in range(self.num_oppo)]
            oppo_num = 0
        for k in self.role_keys:
            if 'teammate' in k:
                team_code_list[team_num] = self.value_team_encoder(self.obs_single_code[k])
                team_key_code_list[team_num] = self.key_team_encoder(self.obs_single_code[k])
                team_num += 1
            if 'opponent' in k:
                oppo_code_list[oppo_num] = self.value_oppo_encoder(self.obs_single_code[k])
                oppo_key_code_list[oppo_num] = self.key_oppo_encoder(self.obs_single_code[k])
                oppo_num += 1
        
        if self.exist_team:
            self.value_team_codes = torch.stack(team_code_list, dim = -2)
            self.key_team_codes = torch.stack(team_key_code_list, dim = -2)
        if self.exist_oppo:
            self.value_oppo_codes = torch.stack(oppo_code_list, dim = -2)
            self.key_oppo_codes = torch.stack(oppo_key_code_list, dim = -2)


    def pair_code(self):
        pair_code_list=[None for _ in range(self.n_pair)]
        pair_key_code_list=[None for _ in range(self.n_pair)]
        pair_temp=0
        for k in self.role_keys:
            if 'proprio' not in k:
                for q in self.role_keys:
                    if 'proprio' not in q and q != k:
                        pair_code_list[pair_temp] = self.value_encoder(torch.cat([self.obs_single_code[k], self.obs_single_code[q]], dim = -1))
                        pair_key_code_list[pair_temp] = self.key_encoder(torch.cat([self.obs_single_code[k], self.obs_single_code[q]], dim = -1))
                        pair_temp+=1
        self.value_codes=torch.stack(pair_code_list, dim=-2)
        self.key_codes=torch.stack(pair_key_code_list, dim=-2)

    def output_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(self.output_size,), dtype=np.float32)

def get_config_test():
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument("--use_ReLU", action='store_false',
                default=True, help="Whether to use ReLU")
    parser.add_argument("--use_feature_normalization", action='store_false',
                        default=True, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--encoder_layer_N",type=int,
                        default=2, help="number of encoder layers")
    parser.add_argument("--encoder_hidden_size", type=int,
                        default=16, help="hidden size of encoder")
    parser.add_argument("--num_agents", type=int, default=4,
                        help="num_agents")
    parser.add_argument("--proprio_shape", type=int, default=13,
                        help="proprio_shape")
    parser.add_argument("--teammate_shape", type=int, default=7,
                        help="teammate")
    parser.add_argument("--opponent_shape", type=int, default=3,
                        help="opponent_shape")
    parser.add_argument("--n_head", type=int, default=4, help="n_head")
    parser.add_argument("--d_k", type=int, default= 8, help="d_k")
    parser.add_argument("--attn_size", type=int, default=20, help="attn_size")

    return parser

if __name__ == "__main__":
    pars=get_config_test()
    args=pars.parse_args()
    print(args)
    obs_mixer_test=Order_Mixer(args, 'pursuer')
    obs_mixer_test_e = Order_Mixer(args, 'evader')
    obs_test=torch.randn(14,37)
    obs_test_e = torch.randn(14,25)
    a=obs_mixer_test(obs_test)
    print("obs=",obs_test.size())
    print(a)
    index=[i for i in range(13)]+[i for i in range(27, 34)] + [i for i in range(20, 27)] + [i for i in range(13, 20)] + [i for i in range(34, 37)]
    index_e = [i for i in range(13)] + [i for i in range(16, 19)] + [i for i in range(22, 25)] + [i for i in range(13, 16)] + [i for i in range(19,22)]
    obs_test_r=obs_test[:,index]
    print("obs_n=",obs_test_r.size())
    b=obs_mixer_test(obs_test_r)
    print(a.size())
    c=a-b
    print(obs_test-obs_test_r)
    print(a-b)
    print(obs_mixer_test.output_space())

    obs_test_e_r = obs_test_e[:, index_e]
    a_e = obs_mixer_test_e(obs_test_e)
    b_e = obs_mixer_test_e(obs_test_e_r)
    print("a_e - b_e = ",a_e-b_e)