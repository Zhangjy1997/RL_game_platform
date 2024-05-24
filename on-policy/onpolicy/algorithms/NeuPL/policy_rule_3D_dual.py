# This script gives the attacker policy based on a 3D rotation-invariant "One VS One" solution
#       effective when 30 < d(attacker, 0) < 200, planar_d(attacker, defender) < 100, and 0 < height < 200
#       attacker's action only depends on the closest defender
# Cost used in the solution: the integration of a running cost, plus a terminal cost
# (1) running cost
#       if attacker is captured or attacker's height goes outside [0, 200], running cost = 0
#       otherwise, running cost is the sum of the following two components
#           (210 - d(attacker, 0))/100 + min((90 - |h(attacker)-100|) / 100, 0.4)
#           ______first component_____   _____________second component______________
#           first component: attacker wants to get closer to the center as much as possible
#           second component: attacker prefers to stay within the height range [50, 150]
# (2) terminal cost
#       if attacker's height goes outside [0, 200], terminal cost = -1000
#       if defender's height goes outside [0, 200], terminal cost = 1000

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interpn

data = loadmat('./mycode/dataforattackerV3.mat')
grid = (np.linspace(25, 205, 46), np.linspace(-100, 100, 51), np.linspace(0, 100, 26),
        np.linspace(-5, 205, 36), np.linspace(-5, 205, 36))
Vm = 7
# state = np.array([attackerx, attackery, attackerz, 
#                   defender1x, defender1y, defender1z,
#                   ..., 
#                   defendernx, defenderny, defendernz])
state = np.array([40., 40., 25., 0., 0., 10., 20., 50., 10.])

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

    # build the 6D reduced state
    reduced_state = state[[0, 1, 2, 3*closest, 3*closest+1, 3*closest+2]]

    # build the 5D rotation-invariant state
    #       ri_state[0]: 1D distance between the attacker and the center
    #       ri_state[1:3]: 2D relative position of the attacker with respect to the defender
    #       ri_state[3]: 1D height of attacker
    #       ri_state[4]: 1D height of defender
    mtx = np.array([[reduced_state[0], reduced_state[1]], 
                    [-reduced_state[1], reduced_state[0]]]) / np.linalg.norm(reduced_state[0:2])
    ri_state = np.concatenate(([np.linalg.norm(reduced_state[0:2])],
                               mtx @ (reduced_state[3:5] - reduced_state[0:2]),
                               reduced_state[[2,5]]))
    
    # ensure PLANAR distance between attacker and defender < 100
    if np.linalg.norm(ri_state[1:3]) > 100:
        ri_state[1:3] = ri_state[1:3] / np.linalg.norm(ri_state[1:3]) * 100
    # ensure ri_state[2] is positive; otherwise, use symmetry
    mark = 1
    if ri_state[2] < 0:
        ri_state[2] = -ri_state[2]
        mark = -1
    
    if ri_state[0] < 200 and ri_state[0] > 30 and \
            ri_state[3] > 0 and ri_state[3] < 200 and \
            ri_state[4] > 0 and ri_state[4] < 200 and \
            np.linalg.norm(np.array([ri_state[1], ri_state[2], ri_state[4]-ri_state[3]])) > 10:
        # if game is still running
        deriv = interpn(grid, data['deriv'], ri_state)[0,:]
        direction = np.array([deriv[0] - deriv[1], 
                              deriv[1] * ri_state[2] / ri_state[0] - deriv[2] * (1 + ri_state[1] / ri_state[0]),
                              deriv[3]])
        # defender policy: 
        # direction = np.array([ -deriv[1], -deriv[2] , -deriv[4]])
        if mark == -1:
            direction[1] = -direction[1]
        direction[0:2] = mtx.T @ direction[0:2]
    else:
        # if game has ended
        direction = np.array([0., 0., 0.])

    # normalization
    if np.linalg.norm(direction) < 1e-4:
        velocity = np.array([0., 0., 0.])
    else:
        velocity = direction / np.linalg.norm(direction) * Vm
    
    return velocity

velocity = attackerpolicy(data, grid, state, Vm)
