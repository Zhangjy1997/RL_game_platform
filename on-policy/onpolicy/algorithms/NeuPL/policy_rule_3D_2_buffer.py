# This script gives the attacker policy based on a 3D "One VS One" solution and a 3D "Two VS One" solution
# 1V1 solution:
#       effective when 20 < d(attacker, 0) < 200, and -120 < x_rel,y_rel,z_rel < 120
#       here, x_rel = x_defender - x_attacker
#       attacker's action only depends on the closest defender
# 2V1 solution:
#       effective when 20 < d(attacker, 0) < 200, and -24 < x_rel,y_rel,z_rel < 24

import numpy as np
from scipy.spatial.distance import cdist
from scipy.io import loadmat
from scipy.interpolate import interpn

data = loadmat('/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattackerV4.mat')
# To save space, the int8 format is used in the mat file
# We need to change data to the float32 format to enable calculation
data['deriv1V1'] = np.single(data['deriv1V1'])
data['deriv2V1'] = np.single(data['deriv2V1'])

grid1V1 = (np.linspace(20, 200, 46), np.linspace(-5, 205, 36),
           np.linspace(-120, 120, 41), np.linspace(-120, 120, 41), np.linspace(-120, 120, 41))
grid2V1 = (np.linspace(20, 200, 19), np.linspace(-5, 205, 15),
           np.linspace(-24, 24, 9), np.linspace(-24, 24, 9), np.linspace(-24, 24, 9),
           np.linspace(-24, 24, 9), np.linspace(-24, 24, 9), np.linspace(-24, 24, 9))
Vm = 7
# state = np.array([attackerx, attackery, attackerz, 
#                   defender1x, defender1y, defender1z,
#                   ..., 
#                   defendernx, defenderny, defendernz])
state = np.array([30., 40., 25., 20., 20., 10., 20., 50., 10.])

def attackerpolicy(data, grid, state, Vm):
    if np.mod(len(state), 3) != 0:
        print('length of state should be a multiple of 3')
        return
    
    # if the two closest defenders satisfy -24 < x_rel,y_rel,z_rel < 24, use 2V1 mode
    numDefenders = int(len(state)/3) - 1
    distance = cdist(state[0:3].reshape(1, 3), state[3:].reshape(numDefenders, 3))[0]
    order = np.argsort(distance) + 1
    mode = '1V1'
    # Here I use a threshold 30 to determine whether to use the 2V1 policy
    if numDefenders >= 2 and \
            np.linalg.norm(state[3*order[0]:3*order[0]+3] - state[0:3]) <= 30 and \
            np.linalg.norm(state[3*order[1]:3*order[1]+3] - state[0:3]) <= 30:
        mode = '2V1'
    
    mtx = np.array([[state[0], state[1]], 
                    [-state[1], state[0]]]) / np.linalg.norm(state[0:2])
    
    if mode == '1V1':
        ri_state = np.concatenate(([np.linalg.norm(state[0:2]), state[2]],
                                   mtx @ (state[3*order[0]:3*order[0]+2] - state[0:2]),
                                   [state[3*order[0]+2] - state[2]]))
        # ensure that ri_state is within the grid range
        for i in range(5):
            if ri_state[i] < grid[0][i][0]:
                ri_state[i] = grid[0][i][0]
            if ri_state[i] > grid[0][i][-1]:
                ri_state[i] = grid[0][i][-1]
        deriv = interpn(grid[0], data['deriv1V1'], ri_state)[0,:]
        direction = np.array([deriv[0] - deriv[2], 
                              deriv[2] * ri_state[3] / ri_state[0] - deriv[3] * (1 + ri_state[2] / ri_state[0]),
                              deriv[1] - deriv[4]])
        direction[0:2] = mtx.T @ direction[0:2]
    elif mode == '2V1':
        ri_state = np.concatenate(([np.linalg.norm(state[0:2]), state[2]],
                                   mtx @ (state[3*order[0]:3*order[0]+ 2] - state[0:2]),
                                   [state[3*order[0]+2] - state[2]],
                                   mtx @ (state[3*order[1]:3*order[1]+ 2] - state[0:2]),
                                   [state[3*order[1]+2] - state[2]]))
        # ensure that ri_state is within the grid range
        for i in range(8):
            if ri_state[i] < grid[1][i][0]:
                ri_state[i] = grid[1][i][0]
            if ri_state[i] > grid[1][i][-1]:
                ri_state[i] = grid[1][i][-1]
        deriv = interpn(grid[1], data['deriv2V1'], ri_state)[0,:]
        direction = np.array([deriv[0] - deriv[2] - deriv[5], 
                              deriv[2] * ri_state[3] / ri_state[0] - deriv[3] * (1 + ri_state[2] / ri_state[0]) +
                              deriv[5] * ri_state[6] / ri_state[0] - deriv[6] * (1 + ri_state[5] / ri_state[0]),
                              deriv[1] - deriv[4] - deriv[7]])
        direction[0:2] = mtx.T @ direction[0:2]

    # normalization
    if np.linalg.norm(direction) < 1e-4:
        velocity = np.array([0., 0., 0.])
    else:
        velocity = direction / np.linalg.norm(direction) * Vm
    
    return velocity

if __name__ == "__main__":
    velocity = attackerpolicy(data, (grid1V1, grid2V1), state, Vm)
    print(velocity)
