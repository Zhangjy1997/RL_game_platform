# This script gives the attacker policy based on a rotation-invariant "One VS One" solution
#       effective when 30 < d(attacker, 0) < 200, and d(attacker, defender) < 100
#       attacker's action only depends on the closest defender
#       attacker never moves along z-direction
# Can be used to
#       compete with existing defender policies
#       train new defender policies (initial z-positions should be set random)
# Cost used in the solution: the integration of a running cost
#       if attacker is captured, running cost = 0
#       otherwise, running cost is always positive, and depends on the distance between attacker and the center

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interpn

# load the reach-avoid set and the derivatives of TTR and TTC
data = loadmat('/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattackerV2.mat')
grid = (np.linspace(10, 210, 101), np.linspace(-100, 100, 101), np.linspace(-100, 100, 101))
Vm = 7  # maximum velocity
# state = np.array([attackerx, attackery, attackerz, 
#                   defender1x, defender1y, defender1z,
#                   ..., 
#                   defendernx, defenderny, defendernz])
state = np.array([40., 40., 5., 0., 0., 0., 60., 20., 0.])

def attackerpolicy(data, grid, state, Vm):
    if np.mod(len(state), 3) != 0:
        print('length of state should be a multiple of 3')
        return
    
    # find the closest defender
    numDefenders = int(len(state)/3) - 1
    closest = 1
    distance = np.linalg.norm(state[3:6] - state[0:3])
    for i in range(2, numDefenders+1):
        new_distance = np.linalg.norm(state[3*i:3*i+3] - state[0:3])
        if new_distance < distance:
            closest = i
            distance = new_distance

    # build the 4D reduced state
    reduced_state = state[[0, 1, 3*closest, 3*closest+1]]

    # build the 3D rotation-invariant state
    #       ri_state[0]: 1D distance between the attacker and the center
    #       ri_state[1:3]: 2D relative position of the attacker with respect to the defender
    mtx = np.array([[reduced_state[0], reduced_state[1]], 
                    [-reduced_state[1], reduced_state[0]]]) / np.linalg.norm(reduced_state[0:2])
    ri_state = np.concatenate(([np.linalg.norm(reduced_state[0:2])],
                               mtx @ (reduced_state[2:4] - reduced_state[0:2])))

    
    # ensure d(attacker, defender) < 100
    if np.linalg.norm(ri_state[1:3]) > 100:
        ri_state[1:3] = ri_state[1:3] / np.linalg.norm(ri_state[1:3]) * 100
    
    if ri_state[0] < 200 and ri_state[0] > 30 and np.linalg.norm(ri_state[1:3]) > 10:
        # if game is still running
        deriv = interpn(grid, data['deriv'], ri_state)[0,:]
        direction = mtx.T @ np.array([deriv[0] - deriv[1], 
                                      deriv[1] * ri_state[2] / ri_state[0] - deriv[2] * (1 + ri_state[1] / ri_state[0])])
    else:
        # if game has ended
        direction = np.array([0., 0.])

    # normalization, and add the 3rd dimension
    if np.linalg.norm(direction) < 1e-4:
        velocity = np.array([0., 0., 0.])
    else:
        velocity = np.append(direction / np.linalg.norm(direction) * Vm, 0.)
    
    return velocity

if __name__ == "__main__":
    velocity = attackerpolicy(data, grid, state, Vm)
    print(velocity)
