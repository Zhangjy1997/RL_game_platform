import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.io import loadmat
from scipy.interpolate import interpn
from onpolicy.algorithms.NeuPL.policy_rule_3D_2_buffer import attackerpolicy as attackerpolicy_3D_2
from scipy.interpolate import RegularGridInterpolator
import time

def reverse_rotation(orientation, position_R):
    rotations = R.from_euler('zyx', orientation)
    inverse_rotation = rotations.inv()
    position_WF = inverse_rotation.apply(position_R)
    return position_WF


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


class evader2lockdown_straight:
    def __init__(self, max_vel, max_heading_rate):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.sim_dt_ = 0.5

    def act(self, obs):
        positions = obs[:, 0:3]
        orientation = obs[:, 3:6]

        position_x, position_y = positions[:, 0:1], positions[:, 1:2]
        horizontal_distance = np.expand_dims(np.linalg.norm(positions[:, :2], axis=1), axis=1)
        sin_theta = -position_y / horizontal_distance
        cos_theta = -position_x / horizontal_distance
        velocity_x = self.max_vel * cos_theta
        velocity_y = self.max_vel * sin_theta
        velocity_z = 0 * cos_theta

        orientation_z = orientation[:, 2:3]
        orientation_target = np.arcsin(sin_theta) # in range [-pi/2, pi/2]
        orientation_target_new = np.where(sin_theta > 0, np.where(cos_theta>0, orientation_target, np.pi - orientation_target), \
            np.where(cos_theta > 0, np.pi - orientation_target, orientation_target + np.pi * 2))

        delta = np.abs((orientation_target_new - orientation_z))
        heading_rate = np.clip(delta, -self.max_heading_rate * self.sim_dt_, self.max_heading_rate * self.sim_dt_) / self.sim_dt_
        evader_action = np.concatenate([velocity_x / 7.0, velocity_y / 7.0, velocity_z / 7.0, heading_rate / 3.14], axis=1)

        return evader_action
    
class Policy_E2L_straight:
    def __init__(self, policy_network, max_vel, max_heading_rate, device):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = evader2lockdown_straight(self.max_vel, self.max_heading_rate)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)
    
class pursuer2evader_straight:
    def __init__(self, max_vel, max_heading_rate, pos_e_inx):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.sim_dt_ = 0.5
        self.pos_e_inx = pos_e_inx

    def act(self, obs):
        positions_p = obs[:, 0:3]
        orientation = obs[:, 3:6]
        #print(orientation)
        positions_e = obs[:, self.pos_e_inx:self.pos_e_inx +3]

        rotations = R.from_euler('zyx', orientation)

        # print(positions_e)

        inverse_rotation = rotations.inv()

        # print(rotations)

        pose_world_frame_delta = inverse_rotation.apply(positions_e)
        # pose_world_frame = positions_p - pose_world_frame_delta
        # print(pose_world_frame)

        delta_position = pose_world_frame_delta

        position_x, position_y, position_z = delta_position[:, 0:1], delta_position[:, 1:2], delta_position[:, 2:3]
        horizontal_distance = np.expand_dims(np.linalg.norm(delta_position, axis=1), axis=1)
        sin_theta = -position_y / horizontal_distance
        cos_theta = -position_x / horizontal_distance
        sin_beta = -position_z / horizontal_distance
        velocity_x = self.max_vel * cos_theta
        velocity_y = self.max_vel * sin_theta
        velocity_z = self.max_vel * sin_beta

        orientation_z = orientation[:, 2:3]
        orientation_target = np.arcsin(sin_theta) # in range [-pi/2, pi/2]
        orientation_target_new = np.where(sin_theta > 0, np.where(cos_theta>0, orientation_target, np.pi - orientation_target), \
            np.where(cos_theta > 0, np.pi - orientation_target, orientation_target + np.pi * 2))

        delta = np.abs((orientation_target_new - orientation_z))
        heading_rate = np.clip(delta, -self.max_heading_rate * self.sim_dt_, self.max_heading_rate * self.sim_dt_) / self.sim_dt_
        pursuer_action = np.concatenate([velocity_x / 7.0, velocity_y / 7.0, velocity_z / 7.0, heading_rate / 3.14], axis=1)
        #print(evader_action)
        return pursuer_action
    
class Policy_P2E_straight:
    def __init__(self, policy_network, max_vel, max_heading_rate, pos_e_inx, device):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.pos_e_inx = pos_e_inx
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = pursuer2evader_straight(self.max_vel, self.max_heading_rate, self.pos_e_inx)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)
    
class evader2Dsuboptimal:
    def __init__(self, max_vel, max_heading, mat_path, samples, pos_p_inx):
        self.max_vel = max_vel
        self.max_heading = max_heading
        self.mat_path = mat_path
        self.data = loadmat(mat_path)
        self.samples = samples
        self.grid = (samples, samples, samples, samples)
        self.pos_p_inx = pos_p_inx
        self.sim_dt_ = 0.5

    def act_raw(self, obs):
        # find the closest defender
        num_defenders = obs.shape[1] // 3 - 1
        states_reshaped = obs.reshape(obs.shape[0], -1, 3)
        attackers = states_reshaped[:, 0, :3]  
        defenders = states_reshaped[:, 1:, :3]  

        # Compute distances from each attacker to all defenders
        distances = np.linalg.norm(defenders - attackers[:, np.newaxis, :], axis=2)
        #print(distances)
        closest_idxs = np.argmin(distances, axis=1)

        closest_defenders = defenders[np.arange(obs.shape[0]), closest_idxs]

        # build the 4D reduced state
        reduced_states = np.concatenate([attackers[:, :2], closest_defenders[:, :2]], axis=1)
        velocities = np.zeros((obs.shape[0], 3))
        norms_raw = np.linalg.norm(reduced_states[:,:2], axis=1)
        # print(norms_raw)
        norms_raw[norms_raw < 1e-4] = 1
        velocities[:,:2] = -reduced_states[:,:2]* self.max_vel / norms_raw[:, np.newaxis]
        mask = np.all(np.linalg.norm(reduced_states.reshape(-1, 2, 2), axis=2) < 100, axis=1)
        # print(np.linalg.norm(reduced_states.reshape(-1, 2, 2), axis=2))
        # print(mask)
        in_range_states = reduced_states[mask]

        ra_values = interpn(self.grid, self.data['RA'], in_range_states, method='linear')
        ttr_directions = interpn(self.grid, self.data['derivTTR'], in_range_states, method='linear')[:, :2]
        ttc_directions = -interpn(self.grid, self.data['derivTTC'], in_range_states, method='linear')[:, :2]

        directions = np.where(ra_values[:, np.newaxis] < 0, ttr_directions, ttc_directions)

        norms = np.linalg.norm(directions, axis=1)
        norms[norms < 1e-4] = 1  # Avoid division by zero
        velocities[mask, :2] = directions * self.max_vel / norms[:, np.newaxis]
        
        return velocities
    
    def act(self, obs):
        positions_e = obs[:, 0:3]
        orientation = obs[:, 3:6]
        positions_p = obs[:, self.pos_p_inx: ]
        rotations = R.from_euler('zyx', orientation)
        inverse_rotation = rotations.inv()
        # print(inverse_rotation.as_matrix())
        pos_long = positions_p.shape[1]
        assert pos_long % 3 ==0, "wrong dimension!"
        split_pos_p = np.split(positions_p, pos_long // 3, axis=1)

        # pos_p_sp = []
        pos_p_wf = []
        # pos_p_sp.append(positions_e)
        pos_p_wf.append(positions_e)
        for i in range(len(split_pos_p)):
            pos_p_wf.append(positions_e - inverse_rotation.apply(split_pos_p[i]))
            # pos_p_sp.append(positions_e - split_pos_p[i])

        position_all_WF = np.concatenate(pos_p_wf, axis=1)
        # position_all_SP = np.concatenate(pos_p_sp, axis=1)

        # print("WF=", position_all_WF)
        # print("SP=", position_all_SP)

        vel_all = self.act_raw(position_all_WF)

        vel_all_y = vel_all[:,1:2]
        vel_all_x = vel_all[:,0:1]

        orientation_z = orientation[:, 2:3]

        orientation_target = np.arctan2(vel_all_y, vel_all_x)
        delta = np.abs((orientation_target - orientation_z))
        heading_rate = np.clip(delta, -self.max_heading * self.sim_dt_, self.max_heading * self.sim_dt_) / self.sim_dt_
        evader_action = np.concatenate([vel_all/7.0 , heading_rate / 3.14], axis=1)
        # evader_random = np.random.uniform(-1, 1, (evader_action.shape[0], evader_action.shape[1]))
        # evader_random[:,2] = 0
        return evader_action

class Policy_E2P_suboptimal:
    def __init__(self, policy_network, max_vel, max_heading_rate, mat_path, samples ,pos_p_inx, device):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.pos_p_inx = pos_p_inx
        self.mat_path = mat_path
        self.samples = samples
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = evader2Dsuboptimal(self.max_vel, self.max_heading_rate, self.mat_path, self.samples, self.pos_p_inx)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)
    
class evader2Doptimal:
    def __init__(self, max_vel, max_heading, mat_path, grid, pos_p_inx):
        self.max_vel = max_vel
        self.max_heading = max_heading
        self.mat_path = mat_path
        self.data = loadmat(mat_path)
        self.grid = grid
        self.pos_p_inx = pos_p_inx
        self.sim_dt_ = 0.5

    def act_raw(self, obs):
        # find the closest defender
        num_defenders = obs.shape[1] // 3 - 1
        states_reshaped = obs.reshape(obs.shape[0], -1, 3)
        attackers = states_reshaped[:, 0, :3]  
        defenders = states_reshaped[:, 1:, :3]  
        # Compute distances from each attacker to all defenders
        distances = np.linalg.norm(defenders - attackers[:, np.newaxis, :], axis=2)
        #print(distances)
        # probs = np.exp(-distances/1.0)
        # probs /= probs.sum(axis=1)[:, np.newaxis]
        # selected_indices = [np.random.choice(len(row), p=row) for row in probs]
        closest_idxs = np.argmin(distances, axis=1)
        closest_defenders = defenders[np.arange(obs.shape[0]), closest_idxs]
        closest_distance = distances[np.arange(obs.shape[0]), closest_idxs]
        reduced_states = np.concatenate([attackers[:, :2], closest_defenders[:, :2]], axis=1)

        norms = np.linalg.norm(reduced_states[:, :2], axis=1)

        mtx = np.array([[reduced_states[:, 0], reduced_states[:, 1]], 
                        [-reduced_states[:, 1], reduced_states[:, 0]]]) / norms
        mtx = np.transpose(mtx, (2, 0, 1))

        rotated_coords = np.einsum('ijk,ik->ij', mtx, reduced_states[:, 2:4] - reduced_states[:, :2])

        ri_states = np.hstack((norms[:, np.newaxis], rotated_coords))

        distances_2D = np.linalg.norm(ri_states[:, 1:3], axis=1)
        mask = distances_2D > 100
        ri_states[mask, 1:3] = ri_states[mask, 1:3] / distances_2D[mask, np.newaxis] * 100

        distances_2D = np.linalg.norm(ri_states[:, 1:3], axis=1)

        mask = (distances_2D < 10) & (closest_distance >10)

        # print("mask=", mask)
        ri_states[mask, 1:3] = ri_states[mask, 1:3] / distances_2D[mask, np.newaxis] * 12

        distances_2D = np.linalg.norm(ri_states[:, 1:3], axis=1)

        game_running = (ri_states[:, 0] < 200) & (ri_states[:, 0] > 30) & (distances_2D > 10)
        directions = np.zeros((ri_states.shape[0], 2))
        
        # derivs = interpn(self.grid, self.data['deriv'], ri_states[game_running], bounds_error=False, fill_value=None)

        for i, ri_state in enumerate(ri_states):
            if game_running[i]:
                deriv = interpn(self.grid, self.data['deriv'], ri_state)[0, :]
                direction = mtx[i].T @ np.array([deriv[0] - deriv[1], 
                                                deriv[1] * ri_state[2] / ri_state[0] - deriv[2] * (1 + ri_state[1] / ri_state[0])])
                directions[i] = direction


        norms_dir = np.linalg.norm(directions, axis=1)

        too_small = norms_dir < 1e-4

        velocities = np.zeros((directions.shape[0], 3))
        
        velocities[~too_small, :2] = directions[~too_small] / norms_dir[~too_small, np.newaxis] * self.max_vel

        return velocities
    
    def act(self, obs):
        positions_e = obs[:, 0:3]
        orientation = obs[:, 3:6]
        positions_p = obs[:, self.pos_p_inx: ]
        rotations = R.from_euler('zyx', orientation)
        inverse_rotation = rotations.inv()
        pos_long = positions_p.shape[1]
        assert pos_long % 3 ==0, "wrong dimension!"
        split_pos_p = np.split(positions_p, pos_long // 3, axis=1)

        pos_p_wf = []

        pos_p_wf.append(positions_e)
        for i in range(len(split_pos_p)):
            pos_p_wf.append(positions_e - inverse_rotation.apply(split_pos_p[i]))
            

        position_all_WF = np.concatenate(pos_p_wf, axis=1)
        # print("WF=", position_all_WF)

        vel_all = self.act_raw(position_all_WF)

        vel_all_y = vel_all[:,1:2]
        vel_all_x = vel_all[:,0:1]

        orientation_z = orientation[:, 2:3]

        orientation_target = np.arctan2(vel_all_y, vel_all_x)
        delta = np.abs((orientation_target - orientation_z))
        heading_rate = np.clip(delta, -self.max_heading * self.sim_dt_, self.max_heading * self.sim_dt_) / self.sim_dt_
        evader_action = np.concatenate([vel_all/7.0 , heading_rate / 3.14], axis=1)
        # print(evader_action)
        return evader_action

class Policy_E2P_optimal:
    def __init__(self, policy_network, max_vel, max_heading_rate, mat_path, grid ,pos_p_inx, device):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.pos_p_inx = pos_p_inx
        self.mat_path = mat_path
        self.grid = grid
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = evader2Doptimal(self.max_vel, self.max_heading_rate, self.mat_path, self.grid, self.pos_p_inx)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)
    
class evader3Doptimal:
    def __init__(self, max_vel, max_heading, mat_path, grid, pos_p_inx):
        self.max_vel = max_vel
        self.max_heading = max_heading
        self.mat_path = mat_path
        self.data = loadmat(mat_path)
        self.grid = grid
        self.pos_p_inx = pos_p_inx
        self.sim_dt_ = 0.5

    def act_raw(self, obs):
         # find the closest defender
        num_defenders = obs.shape[1] // 3 - 1
        states_reshaped = obs.reshape(obs.shape[0], -1, 3)
        attackers = states_reshaped[:, 0, :3]  
        defenders = states_reshaped[:, 1:, :3]
        # Compute distances from each attacker to all defenders
        distances = np.linalg.norm(defenders - attackers[:, np.newaxis, :], axis=2)
        #print(distances)
        closest_idxs = np.argmin(distances, axis=1)
        closest_defenders = defenders[np.arange(obs.shape[0]), closest_idxs]
        closest_distance = distances[np.arange(obs.shape[0]), closest_idxs]
        reduced_states = np.concatenate([attackers[:, :3], closest_defenders[:, :3]], axis=1)

        norms = np.linalg.norm(reduced_states[:, 0:2], axis=1)
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
            # if np.linalg.norm(ri_state[1:3]) > 100:
            #     ri_state[1:3] = ri_state[1:3] / np.linalg.norm(ri_state[1:3]) * 100
            # mark = -1 if ri_state[2] < 0 else 1
            # ri_state[2] = np.abs(ri_state[2])

            # valid_condition = (ri_state[0] < 200) & (ri_state[0] > 30) & \
            #                 (ri_state[3] > 0) & (ri_state[3] < 200) & \
            #                 (ri_state[4] > 0) & (ri_state[4] < 200) & \
            #                 (np.linalg.norm(np.array([ri_state[1], ri_state[2], ri_state[4]-ri_state[3]])) > 10)

            if valid_conditions[i]:
                # start_time_sub = time.time()
                # deriv = interpn(self.grid, self.data['deriv'], ri_state)[0,:]
                # deriv = interpolator(ri_state)[0,:]
                deriv = derivs[i]
                # end_time_sub = time.time()
                # time_a += end_time_sub - start_time_sub
                direction = np.array([deriv[0] - deriv[1], 
                                    deriv[1] * ri_state[2] / ri_state[0] - deriv[2] * (1 + ri_state[1] / ri_state[0]),
                                    deriv[3]])
                direction[1] *= marks[i]
                direction[0:2] = mtx[i].T @ direction[0:2]
                if np.linalg.norm(direction) >= 1e-4:
                    velocities[i] = direction / np.linalg.norm(direction) * self.max_vel

        # print("sub_time = ", time_a)
        return velocities
    
    def act(self, obs):
        positions_e = obs[:, 0:3]
        orientation = obs[:, 3:6]
        positions_p = obs[:, self.pos_p_inx: ]
        rotations = R.from_euler('zyx', orientation)
        inverse_rotation = rotations.inv()
        pos_long = positions_p.shape[1]
        assert pos_long % 3 ==0, "wrong dimension!"
        split_pos_p = np.split(positions_p, pos_long // 3, axis=1)

        pos_p_wf = []

        pos_p_wf.append(positions_e)
        for i in range(len(split_pos_p)):
            pos_p_wf.append(positions_e - inverse_rotation.apply(split_pos_p[i]))
            

        position_all_WF = np.concatenate(pos_p_wf, axis=1)
        # print("WF=", position_all_WF)

        vel_all = self.act_raw(position_all_WF)

        vel_all_y = vel_all[:,1:2]
        vel_all_x = vel_all[:,0:1]

        orientation_z = orientation[:, 2:3]

        orientation_target = np.arctan2(vel_all_y, vel_all_x)
        delta = np.abs((orientation_target - orientation_z))
        heading_rate = np.clip(delta, -self.max_heading * self.sim_dt_, self.max_heading * self.sim_dt_) / self.sim_dt_
        evader_action = np.concatenate([vel_all/7.0 , heading_rate / 3.14], axis=1)
        # print(evader_action)
        return evader_action
    
class Policy_E2P_3Doptimal:
    def __init__(self, policy_network, max_vel, max_heading_rate, mat_path, grid ,pos_p_inx, device):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.pos_p_inx = pos_p_inx
        self.mat_path = mat_path
        self.grid = grid
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = evader3Doptimal(self.max_vel, self.max_heading_rate, self.mat_path, self.grid, self.pos_p_inx)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)
    
class evader_3Dvs2_optimal:
    def __init__(self, max_vel, max_heading, mat_path, grid, pos_p_inx):
        self.max_vel = max_vel
        self.max_heading = max_heading
        self.mat_path = mat_path
        self.data = loadmat(mat_path)
        self.data['deriv1V1'] = np.single(self.data['deriv1V1'])
        self.data['deriv2V1'] = np.single(self.data['deriv2V1'])
        self.grid = grid
        self.pos_p_inx = pos_p_inx
        self.sim_dt_ = 0.5

    def act_raw(self, obs):
        velocities = np.zeros((len(obs),3))
        for i in range(len(obs)):
            velocities[i] = attackerpolicy_3D_2(self.data, self.grid, obs[i], self.max_vel)

        return velocities
    
    def act(self, obs):
        positions_e = obs[:, 0:3]
        orientation = obs[:, 3:6]
        positions_p = obs[:, self.pos_p_inx: ]
        rotations = R.from_euler('zyx', orientation)
        inverse_rotation = rotations.inv()
        pos_long = positions_p.shape[1]
        assert pos_long % 3 ==0, "wrong dimension!"
        split_pos_p = np.split(positions_p, pos_long // 3, axis=1)

        pos_p_wf = []

        pos_p_wf.append(positions_e)
        for i in range(len(split_pos_p)):
            pos_p_wf.append(positions_e - inverse_rotation.apply(split_pos_p[i]))
            

        position_all_WF = np.concatenate(pos_p_wf, axis=1)
        # print("WF=", position_all_WF)

        vel_all = self.act_raw(position_all_WF)

        vel_all_y = vel_all[:,1:2]
        vel_all_x = vel_all[:,0:1]

        orientation_z = orientation[:, 2:3]

        orientation_target = np.arctan2(vel_all_y, vel_all_x)
        delta = np.abs((orientation_target - orientation_z))
        heading_rate = np.clip(delta, -self.max_heading * self.sim_dt_, self.max_heading * self.sim_dt_) / self.sim_dt_
        evader_action = np.concatenate([vel_all/7.0 , heading_rate / 3.14], axis=1)
        # print(evader_action)
        return evader_action
    
class Policy_E2P_3D_2_optimal:
    def __init__(self, policy_network, max_vel, max_heading_rate, mat_path, grid ,pos_p_inx, device):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.pos_p_inx = pos_p_inx
        self.mat_path = mat_path
        self.grid = grid
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = evader_3Dvs2_optimal(self.max_vel, self.max_heading_rate, self.mat_path, self.grid, self.pos_p_inx)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)
    
class pursuer3Doptimal:
    def __init__(self, max_vel, max_heading, mat_path, grid, pos_e_inx):
        self.max_vel = max_vel
        self.max_heading = max_heading
        self.mat_path = mat_path
        self.data = loadmat(mat_path)
        self.grid = grid
        self.pos_e_inx = pos_e_inx
        self.sim_dt_ = 0.5

    def act_raw(self, obs):
        pos_p_self = obs[:, :3]
        num_defenders = obs.shape[1] // 3 - 1
        states_reshaped = obs.reshape(obs.shape[0], -1, 3)
        attackers = obs[:, self.pos_e_inx:self.pos_e_inx + 3]  
        defenders = obs[:, :3]
        reduced_states = np.concatenate([attackers[:, :3], defenders[:, :3]], axis=1)

        norms = np.linalg.norm(reduced_states[:, 0:2], axis=1)
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

        print("mat shape =", self.data['deriv'].shape)
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
            # if np.linalg.norm(ri_state[1:3]) > 100:
            #     ri_state[1:3] = ri_state[1:3] / np.linalg.norm(ri_state[1:3]) * 100
            # mark = -1 if ri_state[2] < 0 else 1
            # ri_state[2] = np.abs(ri_state[2])

            # valid_condition = (ri_state[0] < 200) & (ri_state[0] > 30) & \
            #                 (ri_state[3] > 0) & (ri_state[3] < 200) & \
            #                 (ri_state[4] > 0) & (ri_state[4] < 200) & \
            #                 (np.linalg.norm(np.array([ri_state[1], ri_state[2], ri_state[4]-ri_state[3]])) > 10)

            if valid_conditions[i]:
                # start_time_sub = time.time()
                # deriv = interpn(self.grid, self.data['deriv'], ri_state)[0,:]
                # deriv = interpolator(ri_state)[0,:]
                deriv = derivs[i]
                # end_time_sub = time.time()
                # time_a += end_time_sub - start_time_sub
                # direction = np.array([deriv[0] - deriv[1], 
                #                     deriv[1] * ri_state[2] / ri_state[0] - deriv[2] * (1 + ri_state[1] / ri_state[0]),
                #                     deriv[3]])
                direction = np.array([ -deriv[1], -deriv[2] , -deriv[4]])
                direction[1] *= marks[i]
                direction[0:2] = mtx[i].T @ direction[0:2]
                if np.linalg.norm(direction) >= 1e-4:
                    velocities[i] = direction / np.linalg.norm(direction) * self.max_vel

        # print("sub_time = ", time_a)
        return velocities
    
    def act(self, obs):
        positions_p = obs[:, 0:3]
        orientation = obs[:, 3:6]
        positions_e = obs[:, self.pos_e_inx:self.pos_e_inx +3]

        rotations = R.from_euler('zyx', orientation)

        # print(positions_e)

        inverse_rotation = rotations.inv()

        # print(rotations)

        pose_world_frame_delta = inverse_rotation.apply(positions_e)
        # pose_world_frame = positions_p - pose_world_frame_delta
        # print(pose_world_frame)

        delta_position = pose_world_frame_delta

        pos_e_wf = []
        pos_e_wf.append(positions_p)
        pos_e_wf.append(positions_p - delta_position)
            

        position_all_WF = np.concatenate(pos_e_wf, axis=-1)
        # print("WF=", position_all_WF)

        vel_all = self.act_raw(position_all_WF)

        vel_all_y = vel_all[:,1:2]
        vel_all_x = vel_all[:,0:1]

        orientation_z = orientation[:, 2:3]

        orientation_target = np.arctan2(vel_all_y, vel_all_x)
        delta = np.abs((orientation_target - orientation_z))
        heading_rate = np.clip(delta, -self.max_heading * self.sim_dt_, self.max_heading * self.sim_dt_) / self.sim_dt_
        evader_action = np.concatenate([vel_all/7.0 , heading_rate / 3.14], axis=1)
        # print(evader_action)
        return evader_action
    
class Policy_P2E_3Doptimal:
    def __init__(self, policy_network, max_vel, max_heading_rate, mat_path, grid ,pos_e_inx, device):
        self.max_vel = max_vel
        self.max_heading_rate = max_heading_rate
        self.policy_network = policy_network
        self.actor = self.policy_network.actor
        self.pos_e_inx = pos_e_inx
        self.mat_path = mat_path
        self.grid = grid
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_generator = pursuer3Doptimal(self.max_vel, self.max_heading_rate, self.mat_path, self.grid, self.pos_e_inx)

    def act(self, obs, rnn_state, rnn_mask, available_actions = None, deterministic = True):
        evader_act = self.action_generator.act(check_np(obs))
        evader_act *= check_np(rnn_mask)
        #print(evader_act)
        return check(evader_act).clone().to(**self.tpdv), check(rnn_state).clone().to(**self.tpdv)


if __name__ == "__main__":
    # path = "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattacker.mat"
    # samples = np.linspace(-105, 105, 51)
    # states = np.array([
    #     [40., 40., 5., 0., 0., 0., 60., 50., 0.],
    #     [30., 35., 10., 5., 5., 0., 50., 45., 0.],
    #     [40., 40., 5., 0., 0., 0., 60., 50., 0.]
    #     ])
    
    # random_array = random_float_array = np.random.uniform(-100, 100, (10, 12))
    # print(random_array)

    # evader_policy = evader2Dsuboptimal(max_vel=7, max_heading=0.57/2, mat_path=path, samples=samples, pos_p_inx=13)
    # act_array = evader_policy.act_raw(random_array)
    # print(act_array)
    # for i in range(len(random_array)):
    #     print(act_array[i] - attackerpolicy(evader_policy.data, evader_policy.grid, random_array[i], Vm=7))

    # path = "/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattackerV2.mat"
    # grid = (np.linspace(10, 210, 101), np.linspace(-100, 100, 101), np.linspace(-100, 100, 101))
    path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataV3.mat'
    grid = (np.linspace(25, 205, 46), np.linspace(-100, 100, 51), np.linspace(0, 100, 26),
            np.linspace(-5, 205, 36), np.linspace(-5, 205, 36))
    # path = '/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattackerV4.mat'
    # grid1V1 = (np.linspace(20, 200, 46), np.linspace(-5, 205, 36),
    #        np.linspace(-120, 120, 41), np.linspace(-120, 120, 41), np.linspace(-120, 120, 41))
    # grid2V1 = (np.linspace(20, 200, 19), np.linspace(-5, 205, 15),
    #         np.linspace(-24, 24, 9), np.linspace(-24, 24, 9), np.linspace(-24, 24, 9),
    #         np.linspace(-24, 24, 9), np.linspace(-24, 24, 9), np.linspace(-24, 24, 9))
    # grid = (grid1V1, grid2V1)
    states = np.array([
        [40., 40., 5., 0., 0., 0., 60., 20., 0.],
        [30., 35., 10., 5., 5., 0., 50., 45., 0.],
        [40., 40., 5., 0., 0., 0., 60., 50., 0.]
        ])
    M_states = 32
    N_angents = 15
    random_array = np.random.uniform(0, 200, (M_states, N_angents))
    # print(random_array)
    # evader_policy = evader3Doptimal(max_vel=7, max_heading=0.57/2, mat_path=path, grid = grid, pos_p_inx=13)
    evader_policy = pursuer3Doptimal(max_vel=7, max_heading=0.57/2, mat_path=path, grid = grid, pos_e_inx=13)
    act_array = np.zeros((len(random_array),3))
    total_time1 = 0
    total_time2 = 0
    flag_right = True
    act_array_real = np.empty_like(act_array)
    for st in range(100):
        x = np.random.uniform(-200, 200, (M_states, N_angents))
        y = np.random.uniform(-200, 200, (M_states, N_angents))
        z = np.random.uniform(0, 200, (M_states, N_angents))
        coordinates_3d = np.stack((x, y, z), axis=-1)
        random_array = coordinates_3d.reshape(M_states, 3*N_angents)
        for i in range(len(random_array)):
            start_time1 = time.time()
            # act_array_real[i] = attackerpolicy_3D_2(evader_policy.data, evader_policy.grid, random_array[i], Vm=7)
            end_time1 = time.time()
            total_time1 += end_time1 - start_time1
        start_time2 = time.time()
        act_array = evader_policy.act_raw(random_array)
        end_time2 = time.time()
        total_time2 += end_time2 - start_time2
        if np.all(np.abs(act_array - act_array_real)<1e-10):
            flag_right = True
        else:
            flag_right = False
            break
    print(random_array)
    print(flag_right)
    print(act_array)
    print("delta_act", act_array - act_array_real)
    print("raw_time", total_time1)
    print("v_time", total_time2)

    
