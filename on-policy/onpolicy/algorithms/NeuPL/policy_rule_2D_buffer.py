# This script gives the attacker policy based on 2D "One VS One" TTR and TTC solutions
#       only effective within radius < 100  (data for radius < 200 now generating)
#       attacker's action only depends on the closest defender
#       attacker never moves along z-direction
# Can be used to
#       compete with existing defender policies
#       train new defender policies (initial z-positions should be set random)
# Settings in solving TTR and TTC
#       defenders win, when d(attacker, defender) < 10 for any defender, or when d(attacker, 0) > 100
#       attacker wins, when d(attacker, 0) < 30

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interpn

# load the reach-avoid set and the derivatives of TTR and TTC
data = loadmat("/home/qiyuan/workspace/flightmare_pe/flightrl/on-policy/onpolicy/algorithms/NeuPL/dataforattacker.mat")
samples = np.linspace(-105, 105, 51)
grid = (samples, samples, samples, samples)
Vm = 7  # maximum velocity
# state = np.array([attackerx, attackery, attackerz, 
#                   defender1x, defender1y, defender1z,
#                   ..., 
#                   defendernx, defenderny, defendernz])
state = np.array([40., 40., 5., 0., 0., 0., 60., 50., 0.])

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

    # check that both the defender and the attacker are within range 100
    if np.linalg.norm(reduced_state[0:2]) < 100 and np.linalg.norm(reduced_state[2:4]) < 100:
        if interpn(grid, data['RA'], reduced_state) < 0:
            direction = interpn(grid, data['derivTTR'], reduced_state)[0,:]
        else:
            direction = -interpn(grid, data['derivTTC'], reduced_state)[0,:]
    else:
        direction = -reduced_state[0:2]

    # normalization, and add the 3rd dimension
    if np.linalg.norm(direction) < 1e-4:
        velocity = np.array([0., 0., 0.])
    else:
        velocity = np.append(direction / np.linalg.norm(direction) * Vm, 0.)
    
    return velocity

if __name__ == "__main__":
    velocity = attackerpolicy(data, grid, state, Vm)
    # print(velocity)

    random_array = np.array([
        [43, 88, 70, 11, -85, -60, -108, -33, 10],
        [-104, -116, -7, 52, -131, 140, 143, -112, 92],
        [-86, -99, -23, 144, -88, -1, -14, 0, 60],
        [100, -2, 42, -96, 84, -113, -112, -32, -115],
        [-4, 38, 71, -3, -38, -128, 89, 106, -116],
        [-102, 88, 67, 150, 130, 82, -35, 149, 18],
        [106, -15, 60, 51, -61, 71, 17, 134, 90],
        [38, -62, 88, -72, -144, -55, -101, 98, 15],
        [-17, -90, 41, -105, 43, 120, -86, 114, -37],
        [11, -129, 95, -49, 98, 47, -142, -78, 121]
    ])

    for i in range(len(random_array)):
        print(attackerpolicy(data, grid, random_array[i], Vm))
