import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from flightgym import QuadrotorEnv_v0
from flightgym import MultiQuadrotorEnv_v0
from flightgym import QuadrotorPIDVelCtlEnv_v0
from flightgym import QuadrotorMPCVelCtlEnv_v0
from flightgym import MultiQuadrotorPIDVelCtlEnv_v0
from flightgym import MultiQuadrotorMPCVelCtlEnv_v0
from flightgym import MapStringFloatVector
from flightgym import MapStringFloat
from mpl_toolkits.mplot3d import axes3d
import math

def draw3d():
  # load data from file
  # you replace this using with open
  path = '/home/qiyuan/workspace/flightmare/flightrl/on-policy/onpolicy/scripts/train_uav_scripts'
  data_path = os.path.join(path, '0_observations.npy')
  data = np.load(data_path)
  print("Player0 position:", data[:-1,0:3])
  print("Player1 position:", data[:-1,3:6])

  pur1_x = data[:-1, 0]
  pur1_y = data[:-1, 1]
  pur1_z = data[:-1, 2]

  pur2_x = data[:-1, 3]
  pur2_y = data[:-1, 4]
  pur2_z = data[:-1, 5]

  eva_x = data[:-1, 6]
  eva_y = data[:-1, 7]
  eva_z = data[:-1, 8]

  # checking pursuer move to evader rewards [player0]
  distance_prev = np.sqrt((pur1_x[0:-1] - eva_x[0:-1])**2 + (pur1_y[0:-1] - eva_y[0:-1])**2)
  distance_now = np.sqrt((pur1_x[1:] - eva_x[1:])**2 + (pur1_y[1:] - eva_y[1:])**2)
  progress = (distance_prev - distance_now) * 0.1
  # print(progress)

  # # checking mutual distance
  # distance = np.linalg.norm(data[:-1, 0:3] - data[:-1, 6:9], axis=-1)
  # print("Player0 distance to evader:", distance)

  # distance = np.linalg.norm(data[:-1, 3:6] - data[:-1, 6:9], axis=-1)
  # print("Player1 distance to evader:", distance)

  print("Player speed: ", np.linalg.norm((data[1:-1, 0:3] - data[:-2, 0:3]) / 0.5, axis=-1))

  print("Evader speed: ", np.linalg.norm((data[1:-1, 6:9] - data[:-2, 6:9]) / 0.5, axis=-1))

  # new a figure and set it into 3d
  fig = plt.figure()
  # ax = fig.gca(projection='3d')
  ax = fig.add_subplot(projection = '3d')

  # set figure information
  ax.set_title("3D_Curve")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  # draw the figure, the color is r = read
  figure1 = ax.scatter(pur1_x, pur1_y, pur1_z, c='r')
  figure2 = ax.scatter(pur2_x, pur2_y, pur2_z, c='b')
  figure3 = ax.scatter(eva_x, eva_y, eva_z, c='g')
  plt.show()


if __name__ == "__main__":
    draw3d()
