import numpy as np
import torch

def _t2n(x):
    return x.detach().cpu().numpy()

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output


def check_np(x):
    if isinstance(x, torch.Tensor):
        return _t2n(x)
    else:
        return x

def norm_batch(raw_mat):
    norms = np.linalg.norm(raw_mat, axis=-1)
    mask = norms > 1
    raw_mat[mask, :] = raw_mat[mask, :] / norms[mask, np.newaxis]
    return raw_mat

class Plank2ball_straight:
    def __init__(self, indices, max_vel, max_heading_rate, sim_dt):
        self.indices = indices
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.sim_dt = sim_dt

    def act(self, obs):
        positions = obs[:, 0:2]
        theta = obs[:,2:3]
        pos_ball = obs[:, self.indices]

        norm_vel = norm_batch((pos_ball - positions)/(self.sim_dt * self.max_vel))
        norm_heading = norm_batch((- theta)/(self.sim_dt * self.max_heading_rate))

        evader_action = np.concatenate([norm_vel, norm_heading], axis=-1)

        return evader_action
    
class Policy_P2B_straight:
    def __init__(self, policy_network, max_vel, max_heading_rate, sim_dt, indices, device):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.sim_dt = sim_dt
        self.indices = indices
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = Plank2ball_straight(self.indices, self.max_vel, self.max_heading_rate, self.sim_dt)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)
    
class Plank2ball_rotate:
    def __init__(self, indices, max_vel, max_heading_rate, sim_dt):
        self.indices = indices
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.sim_dt = sim_dt

    def act(self, obs):
        positions = obs[:, 0:2]
        theta = obs[:,2:3]
        pos_ball = obs[:, self.indices]

        norm_vel = norm_batch((pos_ball - positions)/(self.sim_dt * self.max_vel))
        norm_heading = np.zeros_like(norm_vel[:, :1])
        norm_heading[:] = 1

        evader_action = np.concatenate([norm_vel, norm_heading], axis=-1)

        return evader_action
    
class Policy_P2B_rotate:
    def __init__(self, policy_network, max_vel, max_heading_rate, sim_dt, indices, device):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.sim_dt = sim_dt
        self.indices = indices
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = Plank2ball_rotate(self.indices, self.max_vel, self.max_heading_rate, self.sim_dt)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)