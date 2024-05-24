import numpy as np
import torch
import torch.nn as nn
from onpolicy.algorithms.utils.mlp import MLPLayer
from onpolicy.algorithms.utils.attn_moudle import MultiHeadAttention
from scipy.spatial.transform import Rotation as R
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
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        # fix torch

        self.proprio_subsort = torch.arange(self.proprio_shape).unsqueeze(0).to(self.device)
        self.team_subsort = torch.arange(self.teammate_shape).to(self.device)
        self.oppo_subsort = torch.arange(self.opponent_shape).to(self.device)

        if 'pursuer' in self.role_name:
            self.role_keys=['proprioceptive'] + ['teammate_'+str(i) for i in range(self.num_players-1)] + ['opponent_0']
            self.exist_team = True
            self.exist_oppo = True
            self.num_team = self.num_players - 1
            self.num_oppo = 1
        elif 'evader' in self.role_name:
            self.role_keys=['proprioceptive'] + ['opponent_'+str(i) for i in range(self.num_players)]
            self.exist_team = False
            self.exist_oppo = True
            self.num_team = 0
            self.num_oppo = self.num_players
        else:
            print("wrong role name!")

        self.output_size = self.proprio_shape + self.num_team * self.teammate_shape + self.num_oppo * self.opponent_shape

        self.to(device)

    def forward(self, obs):
        positions_p = obs[:, 0:3]
        orientation = obs[:, 3:6]
        rotations = R.from_euler('zyx', orientation)
        inverse_rotation = rotations.inv()
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
        self.obs_role_dict=dict(zip(self.role_keys,obs_role_list))

        self.norm_code()
        index_bias = 0
        init_sort = torch.zeros(obs.shape[0], 1, dtype=torch.int64).to(self.device)
        proprio_index = self.proprio_shape * init_sort + self.proprio_subsort
        index_bias += self.proprio_shape
        self.sort_list = []
        self.sort_list.append(proprio_index)
        if self.exist_team:
            _, team_index = torch.sort(self.value_team_codes, dim = -1)
            # print(team_index[:, 0].shape)
            expanded_index = torch.cat([index_bias + self.teammate_shape * team_index[:, i].unsqueeze(1) + self.team_subsort for i in range(team_index.shape[1])], dim=-1)
            index_bias += self.num_team * self.teammate_shape
            self.sort_list.append(expanded_index)
        if self.exist_oppo:
            _, oppo_index = torch.sort(self.value_oppo_codes, dim = -1)
            # print("oppo_inx = ", oppo_index)

            expanded_index = torch.cat([index_bias + (self.opponent_shape) * oppo_index[:, i].unsqueeze(1) + self.oppo_subsort for i in range(oppo_index.shape[1])], dim=-1)
            self.sort_list.append(expanded_index)
        self.sort_index = torch.cat(self.sort_list, dim = -1)
        # print(self.sort_index)
        obs_out = torch.gather(obs, 1, self.sort_index)
        if 'pursuer' in self.role_name:
            positions_o = [obs_out[:,(self.proprio_shape + i * self.teammate_shape):(self.proprio_shape + i * self.teammate_shape +3)].to('cpu').numpy() for i in range(self.num_players)]
        else:
            positions_o = [obs_out[:,(self.proprio_shape + i * self.opponent_shape):(self.proprio_shape + i * self.opponent_shape +3)].to('cpu').numpy() for i in range(self.num_players)]
        positions_wf = []
        positions_wf.append(positions_p)
        for i in range(self.num_players):
            positions_wf.append(positions_p - inverse_rotation.apply(positions_o[i]))

        position_all_WF = np.concatenate(positions_wf, axis=1)
        position_all_WF = check(position_all_WF).to(**self.tpdv)
        
        return position_all_WF
    
    def norm_code(self):
        if self.exist_team:
            team_code_list = [None for _ in range(self.num_team)]
            team_num = 0
        if self.exist_oppo:
            oppo_code_list = [None for _ in range(self.num_oppo)]
            oppo_num = 0

        for k in self.role_keys:
            if 'teammate' in k:
                team_code_list[team_num] = torch.norm(self.obs_role_dict[k][:,0:3], dim = -1).unsqueeze(-1)
                team_num += 1
            if 'opponent' in k:
                oppo_code_list[oppo_num] = torch.norm(self.obs_role_dict[k][:,0:3], dim = -1).unsqueeze(-1)
                oppo_num += 1

        if self.exist_team:
            self.value_team_codes = torch.cat(team_code_list, dim = -1)
        if self.exist_oppo:
            self.value_oppo_codes = torch.cat(oppo_code_list, dim = -1)
    


    def output_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=((self.num_players+1)*3,), dtype=np.float32)

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
    parser.add_argument("--num_agents", type=int, default=3,
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
    obs_test=torch.randn(14,30)
    obs_test_e = torch.randn(14,22)
    a=obs_mixer_test(obs_test)
    print("obs=",obs_test.size())
    print(a)
    index=[i for i in range(13)] + [i for i in range(20, 27)] + [i for i in range(13, 20)] + [i for i in range(27, 30)]
    index_e = [i for i in range(13)] + [i for i in range(16, 19)] + [i for i in range(13, 16)] + [i for i in range(19,22)]
    obs_test_r=obs_test[:,index]
    print("obs_n=",obs_test_r.size())
    flag_zero = True
    for i in range(100):
        b=obs_mixer_test(obs_test_r)
        if torch.all(torch.abs(a-b) < 1e-10):
            flag_zero = True
        else:
            flag_zero = False
            print(a-b)
            break
    print("flag_zero = ",flag_zero)
    print(a.size())
    c=a-b
    print(obs_test-obs_test_r)
    print("a-b = ", a-b)
    print(obs_mixer_test.output_space())

    obs_test_e_r = obs_test_e[:, index_e]
    a_e = obs_mixer_test_e(obs_test_e)
    b_e = obs_mixer_test_e(obs_test_e_r)
    print("a_e - b_e = ",a_e-b_e)
    print("a_e = ",a_e)
    print("b_e =",obs_test_e)