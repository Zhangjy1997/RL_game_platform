import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat
from scipy.interpolate import interpn
from onpolicy.algorithms.NeuPL.policy_rule_3D_2_buffer import attackerpolicy as attackerpolicy_3D_2
from scipy.interpolate import RegularGridInterpolator
import time

def reverse_clip(value, epsilon):
    if -epsilon < value < epsilon:
        if np.abs(np.sign(value))<0.5:
            return epsilon
        else:
            return np.sign(value) * epsilon
    return value


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


class evader3Doptimal:
    def __init__(self, max_vel, mat_path, grid):
        self.max_vel = max_vel
        # self.max_heading = max_heading
        self.mat_path = mat_path
        self.data = loadmat(mat_path)
        self.grid = grid
        self.sim_dt_ = 0.5

    def act_raw(self, obs):
        # find the closest defender
        num_defenders = obs.shape[1] // 3 - 1
        states_reshaped = obs.reshape(obs.shape[0], -1, 3)
        attackers = states_reshaped[:, 0, :3]  
        defenders = states_reshaped[:, 1:, :3]
        # Compute distances from each attacker to all defenders
        distances = np.linalg.norm(defenders - attackers[:, np.newaxis, :], axis=2)
        
        closest_idxs = np.argmin(distances, axis=1)
        closest_defenders = defenders[np.arange(obs.shape[0]), closest_idxs]
        closest_distance = distances[np.arange(obs.shape[0]), closest_idxs]
        reduced_states = np.concatenate([attackers[:, :3], closest_defenders[:, :3]], axis=1)

        norms = np.linalg.norm(reduced_states[:, 0:2], axis=1) + 1e-8
        mtx = np.array([
            [reduced_states[:, 0], reduced_states[:, 1]],
            [-reduced_states[:, 1], reduced_states[:, 0]]
        ]) / norms
        mtx = np.transpose(mtx, (2, 0, 1))
        
        ri_states = np.concatenate([
            norms[:, np.newaxis],
            np.einsum('ijk,ik->ij', mtx, reduced_states[:, 3:5] - reduced_states[:, 0:2]),
            reduced_states[:, [2, 5]]
        ], axis=1)


        velocities = np.zeros((obs.shape[0], 3))
        # time_a = 0

        interpolator = RegularGridInterpolator(self.grid, self.data['deriv'])

        norms = np.linalg.norm(ri_states[:, 1:3], axis=1)
        mask = norms > 100
        ri_states[mask, 1:3] = ri_states[mask, 1:3] / norms[mask, np.newaxis] * 100
        marks = np.where(ri_states[:, 2] < 0, -1, 1)
        ri_states[:, 2] = np.abs(ri_states[:, 2])

        valid_conditions = (ri_states[:, 0] < 200) & (ri_states[:, 0] > 30) & \
                            (ri_states[:, 3] > 0) & (ri_states[:, 3] < 200) & \
                            (ri_states[:, 4] > 0) & (ri_states[:, 4] < 200) & \
                            (np.linalg.norm(np.stack([ri_states[:, 1], ri_states[:, 2], ri_states[:, 4]-ri_states[:, 3]], axis= 1), axis=1) > 10)
        # print(valid_conditions)
        derivs = np.zeros((obs.shape[0], self.data['deriv'].shape[-1]))
        derivs[valid_conditions] = interpolator(ri_states[valid_conditions])

        for i, ri_state in enumerate(ri_states):

            if valid_conditions[i]:

                deriv = derivs[i]
                direction = np.array([deriv[0] - deriv[1], 
                                    deriv[1] * ri_state[2] / reverse_clip(ri_state[0] , 1e-10) - deriv[2] * (1 + ri_state[1] / reverse_clip(ri_state[0] , 1e-10)),
                                    deriv[3]])
                direction[1] *= marks[i]
                direction[0:2] = mtx[i].T @ direction[0:2]
                if np.linalg.norm(direction) >= 1e-4:
                    velocities[i] = direction / np.linalg.norm(direction) * self.max_vel

        return velocities
    
    def act(self, obs):
        assert obs.shape[1] % 3 ==0, "wrong dimension!"

        vel_all = self.act_raw(obs)

        # heading_rate = np.clip(delta, -self.max_heading * self.sim_dt_, self.max_heading * self.sim_dt_) / self.sim_dt_
        heading_rate = np.zeros((vel_all.shape[0], 1))
        evader_action = np.concatenate([vel_all/7.0 , heading_rate / 3.14], axis=1)
        # print(evader_action)
        return evader_action
    
class Policy_E2P_3Doptimal:
    def __init__(self, policy_network, max_vel, mat_path, grid , device):
        self.max_vel = max_vel
        # self.max_heading_rate = max_heading_rate
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.mat_path = mat_path
        self.grid = grid
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = evader3Doptimal(self.max_vel, self.mat_path, self.grid)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)
    

class pursuer2evader_straight:
    def __init__(self, max_vel, pos_e_inx):
        self.max_vel = max_vel
        self.sim_dt_ = 0.5
        self.pos_e_inx = pos_e_inx

    def act(self, obs):
        positions_p = obs[:, 0:3]
        #print(orientation)
        positions_e = obs[:, self.pos_e_inx:self.pos_e_inx +3]


        delta_position = positions_p - positions_e

        position_x, position_y, position_z = delta_position[:, 0:1], delta_position[:, 1:2], delta_position[:, 2:3]
        horizontal_distance = np.expand_dims(np.linalg.norm(delta_position, axis=1), axis=1)
        sin_theta = -position_y / (horizontal_distance + 1e-10)
        cos_theta = -position_x / (horizontal_distance + 1e-10)
        sin_beta = -position_z / (horizontal_distance + 1e-10)
        velocity_x = self.max_vel * cos_theta
        velocity_y = self.max_vel * sin_theta
        velocity_z = self.max_vel * sin_beta

        heading_rate = np.zeros_like(velocity_x)
        pursuer_action = np.concatenate([velocity_x / 7.0, velocity_y / 7.0, velocity_z / 7.0, heading_rate / 3.14], axis=1)
        #print(evader_action)
        return pursuer_action
    
class Policy_P2E_straight:
    def __init__(self, policy_network, max_vel, pos_e_inx, device):
        self.max_vel = max_vel
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.pos_e_inx = pos_e_inx
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = pursuer2evader_straight(self.max_vel, self.pos_e_inx)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        pursuer_act = self.action_generator.act(check_np(obs))
        pursuer_act *= check_np(rnn_mask)
        # print(pursuer_act)
        return check(pursuer_act).clone().to(**self.tpdv), torch.from_numpy(rnn_state).clone().to(**self.tpdv)
    
class evader2lockdown_straight:
    def __init__(self, max_vel):
        self.max_vel = max_vel

    def act(self, obs):
        positions = obs[:, 0:3]

        position_x, position_y = positions[:, 0:1], positions[:, 1:2]
        horizontal_distance = np.expand_dims(np.linalg.norm(positions[:, :2], axis=1), axis=1)
        sin_theta = -position_y / (horizontal_distance + 1e-10)
        cos_theta = -position_x / (horizontal_distance + 1e-10)
        velocity_x = self.max_vel * cos_theta
        velocity_y = self.max_vel * sin_theta
        velocity_z = 0 * cos_theta

        heading_rate = np.zeros_like(velocity_x)
        evader_action = np.concatenate([velocity_x / self.max_vel, velocity_y / self.max_vel, velocity_z / self.max_vel, heading_rate / 3.14], axis=1)

        return evader_action
    
class Policy_E2L_straight:
    def __init__(self, policy_network, max_vel, device):
        self.max_vel = max_vel
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = evader2lockdown_straight(self.max_vel)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)

class pursuer3Doptimal:
    def __init__(self, max_vel, mat_path, grid):
        self.max_vel = max_vel
        # self.max_heading = max_heading
        self.mat_path = mat_path
        self.data = loadmat(mat_path)
        self.grid = grid
        self.sim_dt_ = 0.5

    def act_raw(self, obs):
        # find the closest defender
        num_defenders = obs.shape[1] // 3 - 1
        states_reshaped = obs.reshape(obs.shape[0], -1, 3)
        attackers = states_reshaped[:, -1, :3]  
        defenders = states_reshaped[:, 0, :3]
        # Compute distances from each attacker to all defenders
        reduced_states = np.concatenate([attackers[:, :3], defenders[:, :3]], axis=1)

        norms = np.linalg.norm(reduced_states[:, 0:2], axis=1) + 1e-8
        mtx = np.array([
            [reduced_states[:, 0], reduced_states[:, 1]],
            [-reduced_states[:, 1], reduced_states[:, 0]]
        ]) / norms
        mtx = np.transpose(mtx, (2, 0, 1))
        
        ri_states = np.concatenate([
            norms[:, np.newaxis],
            np.einsum('ijk,ik->ij', mtx, reduced_states[:, 3:5] - reduced_states[:, 0:2]),
            reduced_states[:, [2, 5]]
        ], axis=1)


        velocities = np.zeros((obs.shape[0], 3))
        # time_a = 0

        interpolator = RegularGridInterpolator(self.grid, self.data['deriv'])

        norms = np.linalg.norm(ri_states[:, 1:3], axis=1)
        mask = norms > 100
        ri_states[mask, 1:3] = ri_states[mask, 1:3] / norms[mask, np.newaxis] * 100
        marks = np.where(ri_states[:, 2] < 0, -1, 1)
        ri_states[:, 2] = np.abs(ri_states[:, 2])

        valid_conditions = (ri_states[:, 0] < 200) & (ri_states[:, 0] > 30) & \
                            (ri_states[:, 3] > 0) & (ri_states[:, 3] < 200) & \
                            (ri_states[:, 4] > 0) & (ri_states[:, 4] < 200) & \
                            (np.linalg.norm(np.stack([ri_states[:, 1], ri_states[:, 2], ri_states[:, 4]-ri_states[:, 3]], axis= 1), axis=1) > 10)
        # print(valid_conditions)
        derivs = np.zeros((obs.shape[0], self.data['deriv'].shape[-1]))
        derivs[valid_conditions] = interpolator(ri_states[valid_conditions])

        for i, ri_state in enumerate(ri_states):

            if valid_conditions[i]:

                deriv = derivs[i]
                # direction = np.array([deriv[0] - deriv[1], 
                #                     deriv[1] * ri_state[2] / reverse_clip(ri_state[0] , 1e-10) - deriv[2] * (1 + ri_state[1] / reverse_clip(ri_state[0] , 1e-10)),
                #                     deriv[3]])
                direction = np.array([ -deriv[1], -deriv[2] , -deriv[4]])
                direction[1] *= marks[i]
                direction[0:2] = mtx[i].T @ direction[0:2]
                if np.linalg.norm(direction) >= 1e-4:
                    velocities[i] = direction / np.linalg.norm(direction) * self.max_vel

        return velocities
    
    def act(self, obs):
        assert obs.shape[1] % 3 ==0, "wrong dimension!"

        vel_all = self.act_raw(obs)

        # heading_rate = np.clip(delta, -self.max_heading * self.sim_dt_, self.max_heading * self.sim_dt_) / self.sim_dt_
        heading_rate = np.zeros((vel_all.shape[0], 1))
        evader_action = np.concatenate([vel_all/7.0 , heading_rate / 3.14], axis=1)
        # print(evader_action)
        return evader_action
    
class Policy_P2E_3Doptimal:
    def __init__(self, policy_network, max_vel, mat_path, grid , device):
        self.max_vel = max_vel
        # self.max_heading_rate = max_heading_rate
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.mat_path = mat_path
        self.grid = grid
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = pursuer3Doptimal(self.max_vel, self.mat_path, self.grid)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)