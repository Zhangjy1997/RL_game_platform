import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os



def plot_track(p_l,e_l):
    n=len(p_l)
    m1=len(p_l[0])
    m2=len(e_l[0])
    x=np.zeros((m1+m2,n))
    y=np.zeros((m1+m2,n))
    z=np.zeros((m1+m2,n))
    for i in range(n):
        for j in range(m1):
            x[j][i]=p_l[i][j][0]
            y[j][i]=p_l[i][j][1]
            z[j][i]=p_l[i][j][2]
        for j in range(m2):
            x[m1 + j][i]=e_l[i][j][0]
            y[m1 + j][i]=e_l[i][j][1]
            z[m1 + j][i]=e_l[i][j][2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(m1):
        ax.plot(x[i], y[i], z[i], c='r')
    for i in range(m2):
        ax.plot(x[m1 + i], y[m1 + i], z[m1 + i], c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_zlim(0,50)
    plt.show()

def plot_git(args, p_l, e_l):
    base_filename = 'frame_'

    n=len(p_l)
    m1=len(p_l[0])
    m2=len(e_l[0])
    x=np.zeros((m1+m2,n))
    y=np.zeros((m1+m2,n))
    z=np.zeros((m1+m2,n))
    for i in range(n):
        for j in range(m1):
            x[j][i]=p_l[i][j][0]
            y[j][i]=p_l[i][j][1]
            z[j][i]=p_l[i][j][2]
        for j in range(m2):
            x[m1 + j][i]=e_l[i][j][0]
            y[m1 + j][i]=e_l[i][j][1]
            z[m1 + j][i]=e_l[i][j][2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    filenames = []
    
    for i in range(n):
        ax.clear()
        for j in range(m1):
            # ax.plot(x[j,0:i], y[j,0:i], z[j,0:i], c='b')
            for k in range(1, i):
                alpha = min(max(0.05, np.exp(-0.2*(i - k - 5))),1)
                ax.plot(x[j, k-1:k+1], y[j, k-1:k+1], z[j, k-1:k+1], c='b', alpha=alpha)

        for j in range(m2):
            # ax.plot(x[m1 + j, 0:i], y[m1 + j, 0:i], z[m1 + j, 0:i], c='r')
            for k in range(1, i):
                alpha = min(max(0.05, np.exp(-0.2*(i - k - 5))),1)
                ax.plot(x[m1 + j, k-1:k+1], y[m1 + j, k-1:k+1], z[m1 + j, k-1:k+1], c='r', alpha=alpha)
        #ax.scatter(position[0], position[1], position[2])

        ax.set_xlim([np.min(x), np.max(x)])
        ax.set_ylim([np.min(y), np.max(y)])
        ax.set_zlim([np.min(z), np.max(z)])
        plt.title("pursuer_" + str(args.pursuer_num) + " vs evader_" + str(args.evader_num))

        filename = os.path.join(args.file_path, f'{base_filename}{i}.png')
        plt.savefig(filename)
        filenames.append(filename)

    i = 1
    while os.path.isfile(os.path.join(args.file_path, f"{args.file_name}{i}.gif")):
        i += 1
    next_filename = os.path.join(args.file_path, f"{args.file_name}{i}.gif")
    with imageio.get_writer(next_filename, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)

def get_track_gif(p_l, e_l, save_path, save_filename, track_title):
    base_filename = 'frame_'

    n=len(p_l)
    m1=len(p_l[0])
    m2=len(e_l[0])
    x=np.zeros((m1+m2,n))
    y=np.zeros((m1+m2,n))
    z=np.zeros((m1+m2,n))
    for i in range(n):
        for j in range(m1):
            x[j][i]=p_l[i][j][0]
            y[j][i]=p_l[i][j][1]
            z[j][i]=p_l[i][j][2]
        for j in range(m2):
            x[m1 + j][i]=e_l[i][j][0]
            y[m1 + j][i]=e_l[i][j][1]
            z[m1 + j][i]=e_l[i][j][2]

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(90, 150, figure=fig)
    ax_main = fig.add_subplot(gs[:,0:115], projection='3d')  
    ax_top = fig.add_subplot(gs[3:27,123:147])  
    ax_front = fig.add_subplot(gs[33:57,123:147])  
    ax_side = fig.add_subplot(gs[63:87,123:147])  
    filenames = []
    
    for i in range(n):
        ax_main.clear()
        ax_top.clear()
        ax_front.clear()
        ax_side.clear()

        for j in range(m1):
            # ax.plot(x[j,0:i], y[j,0:i], z[j,0:i], c='b')
            for k in range(1, i):
                alpha = min(max(0.05, np.exp(-0.2*(i - k - 5))),1)
                ax_main.plot(x[j, k-1:k+1], y[j, k-1:k+1], z[j, k-1:k+1], c='b', alpha=alpha)

        for j in range(m2):
            # ax.plot(x[m1 + j, 0:i], y[m1 + j, 0:i], z[m1 + j, 0:i], c='r')
            for k in range(1, i):
                alpha = min(max(0.05, np.exp(-0.2*(i - k - 5))),1)
                ax_main.plot(x[m1 + j, k-1:k+1], y[m1 + j, k-1:k+1], z[m1 + j, k-1:k+1], c='r', alpha=alpha)
        #ax.scatter(position[0], position[1], position[2])

        # lockdown zone
        radius = 30  
        height = np.max(z) - np.min(z)  
        z_center = (np.max(z) + np.min(z)) / 2  
        theta = np.linspace(0, 2*np.pi, 100)  
        z_line = np.linspace(z_center - height / 2, z_center + height / 2, 100) 
        theta, z_line = np.meshgrid(theta, z_line)  
        x_cylinder = radius * np.cos(theta)  
        y_cylinder = radius * np.sin(theta)  
        z_cylinder = z_line  

        ax_main.plot_surface(x_cylinder, y_cylinder, z_cylinder, color='g', alpha=0.3)

        max_r = max(np.max(x)-np.min(x), np.max(y)-np.min(y))
        c_x = (np.max(x)+np.min(x))/2
        c_y = (np.max(y)+np.min(y))/2

        ax_main.set_xlim([c_x-max_r/2, c_x+max_r/2])
        ax_main.set_ylim([c_y-max_r/2, c_y+max_r/2])
        ax_main.set_zlim([np.min(z), np.max(z)])

        ax_top.set_xlim([c_x-max_r/2, c_x+max_r/2])
        ax_top.set_ylim([c_y-max_r/2, c_y+max_r/2])

        ax_front.set_xlim([c_x-max_r/2, c_x+max_r/2])
        ax_front.set_ylim([np.min(z), np.max(z)])

        ax_side.set_xlim([c_y-max_r/2, c_y+max_r/2])
        ax_side.set_ylim([np.min(z), np.max(z)])

        # plt.title(track_title)
        ax_main.set_title(track_title)
        ax_top.set_title('Top View')
        ax_front.set_title('Front View')
        ax_side.set_title('Side View')

        for j in range(m1):
            ax_top.plot(x[j,:i], y[j,:i], c='b')
            ax_front.plot(x[j,:i], z[j,:i], c='b')
            ax_side.plot(y[j,:i], z[j,:i], c='b')
        for j in range(m2):
            ax_top.plot(x[j+m1,:i], y[j+m1,:i], c='r')
            ax_front.plot(x[j+m1,:i], z[j+m1,:i], c='r')
            ax_side.plot(y[j+m1,:i], z[j+m1,:i], c='r')

        circle = plt.Circle((0, 0), radius, color='g', alpha=0.3)
        ax_top.add_artist(circle)

        ax_front.fill_betweenx([np.min(z), np.max(z)], -radius, radius, color='g', alpha=0.3)
        ax_side.fill_betweenx([np.min(z), np.max(z)], -radius, radius, color='g', alpha=0.3)

        filename = os.path.join(save_path, f'{base_filename}{i}.png')
        plt.savefig(filename)
        filenames.append(filename)
    
    plt.close(fig)

    i = 1
    while os.path.isfile(os.path.join(save_path, f"{save_filename}{i}.gif")):
        i += 1
    next_filename = os.path.join(save_path, f"{save_filename}{i}.gif")
    with imageio.get_writer(next_filename, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)

def plot_gif_plankNball(file_path, file_name, plank1_state, plank2_state, ball_state, env_state):
    base_filename = 'frame_'

    plank1_pos = plank1_state['position']
    plank1_theta = plank1_state['orientation']
    plank1_length = plank1_state['plank_length']

    plank1_degree = np.degrees(plank1_theta)

    plank2_pos = plank2_state['position']
    plank2_theta = plank2_state['orientation']
    plank2_length = plank2_state['plank_length']

    plank2_degree = np.degrees(plank2_theta)

    ball_pos = ball_state['position']

    world_box = env_state['world_box']
    lose_zone_x = env_state['lose_zone_x']
    mid_zone_x = env_state['mid_zone_x']


    n=len(plank1_pos)

    reduce_plank1 = np.zeros((n,6))
    # reduce_plank1[:, 4:] = plank1_degree
    # reduce_plank1[:, 3:4] = plank1_state['plank_height']
    # reduce_plank1[:, 2:3] = plank1_length

    sub_height = 1
    res_height_1 = max(plank1_state['plank_height']-sub_height, 0)
    res_height_2 = max(plank2_state['plank_height']-sub_height, 0)

    plank1_base_point = plank1_pos + plank1_length/2 * np.concatenate([-np.cos(plank1_theta), -np.sin(plank1_theta)], axis=-1)
    plank1_base_point_1 = plank1_base_point + plank1_state['plank_height'] * np.concatenate([np.sin(plank1_theta), -np.cos(plank1_theta)], axis=-1)
    plank1_base_point_2 = plank1_base_point + sub_height * np.concatenate([np.sin(plank1_theta), -np.cos(plank1_theta)], axis=-1)
    plank1_base_point_3 = plank1_base_point - sub_height * np.concatenate([np.sin(plank1_theta), -np.cos(plank1_theta)], axis=-1)
    reduce_plank1[:, :2] = plank1_base_point_1
    reduce_plank1[:, 2:4] = plank1_base_point_2
    reduce_plank1[:, 4:] = plank1_base_point_3
    

    reduce_plank2 = np.zeros((n,6))
    # reduce_plank2[:, 4:] = plank2_degree
    # reduce_plank2[:, 3:4] = plank2_state['plank_height']
    # reduce_plank2[:, 2:3] = plank2_length

    plank2_base_point = plank2_pos + plank2_length/2 * np.concatenate([-np.cos(plank2_theta), -np.sin(plank2_theta)], axis=-1)
    plank2_base_point_1 = plank2_base_point + plank2_state['plank_height'] * np.concatenate([np.sin(plank2_theta), -np.cos(plank2_theta)], axis=-1)
    plank2_base_point_2 = plank2_base_point + sub_height * np.concatenate([np.sin(plank2_theta), -np.cos(plank2_theta)], axis=-1)
    plank2_base_point_3 = plank2_base_point - sub_height * np.concatenate([np.sin(plank2_theta), -np.cos(plank2_theta)], axis=-1)
    reduce_plank2[:, :2] = plank2_base_point_1
    reduce_plank2[:, 2:4] = plank2_base_point_2
    reduce_plank2[:, 4:] = plank2_base_point_3

    fig, ax = plt.subplots(figsize=(6, 16))
    reduce_world_box = [world_box[2], world_box[0], world_box[3]-world_box[2], world_box[1]-world_box[0]]
    field = patches.Rectangle((reduce_world_box[0], reduce_world_box[1]), reduce_world_box[2], reduce_world_box[3], linewidth=2, edgecolor='k', facecolor='none')
    filenames = []
    
    for i in range(n):
        ax.clear()
        ax.add_patch(field)
        tiny_plank = [
            {'xy': (reduce_plank1[i,2], reduce_plank1[i,3]), 'width': plank1_length, 'height': 2*sub_height, 'angle': plank1_degree[i,0]},
            {'xy': (reduce_plank2[i,2], reduce_plank2[i,3]), 'width': plank2_length, 'height': 2*sub_height, 'angle': plank2_degree[i,0]},
            ]
        
        full_plank = [
            {'xy': (reduce_plank1[i,0], reduce_plank1[i,1]), 'width': plank1_length, 'height': res_height_1, 'angle': plank1_degree[i,0]},
            {'xy': (reduce_plank1[i,4], reduce_plank1[i,5]), 'width': plank1_length, 'height': res_height_1, 'angle': plank1_degree[i,0]},
            {'xy': (reduce_plank2[i,0], reduce_plank2[i,1]), 'width': plank2_length, 'height': res_height_2, 'angle': plank2_degree[i,0]},
            {'xy': (reduce_plank2[i,4], reduce_plank2[i,5]), 'width': plank2_length, 'height': res_height_2, 'angle': plank2_degree[i,0]},
            ]
        
        lose_zone_list=[
            {'xy': (reduce_world_box[0], reduce_world_box[1]), 'width': reduce_world_box[2], 'height': (lose_zone_x[0] - world_box[0]), 'angle': 0},
            {'xy': (reduce_world_box[0], lose_zone_x[1]), 'width': reduce_world_box[2], 'height': (world_box[1] - lose_zone_x[1]), 'angle': 0},
        ]

        player_zone_list=[
            {'xy': (reduce_world_box[0], mid_zone_x[1]), 'width': reduce_world_box[2], 'height': (lose_zone_x[1] - mid_zone_x[1]), 'angle': 0},
            {'xy': (reduce_world_box[0], lose_zone_x[0]), 'width': reduce_world_box[2], 'height': (mid_zone_x[0] - lose_zone_x[0]), 'angle': 0},
        ]

        mid_zone_list=[
            {'xy': (reduce_world_box[0], mid_zone_x[0]), 'width': reduce_world_box[2], 'height': (mid_zone_x[1] - mid_zone_x[0]), 'angle': 0},
        ]


        balls = [
            {'xy': (ball_pos[i,0], ball_pos[i,1]), 'radius': ball_state['ball_radius']},
            ]
        
        for zone in lose_zone_list:
            patch = patches.Rectangle(zone['xy'], zone['width'], zone['height'], angle=zone['angle'], linewidth=5, edgecolor='k', facecolor='#54ABFA', alpha=1, transform=ax.transData)
            ax.add_patch(patch)

        for zone in player_zone_list:
            patch = patches.Rectangle(zone['xy'], zone['width'], zone['height'], angle=zone['angle'], linewidth=5, edgecolor='k', facecolor='#92D050', alpha=1, transform=ax.transData)
            ax.add_patch(patch)

        for zone in mid_zone_list:
            patch = patches.Rectangle(zone['xy'], zone['width'], zone['height'], angle=zone['angle'], linewidth=5, edgecolor='k', facecolor='#FDFDEB', alpha=1, transform=ax.transData)
            ax.add_patch(patch)

        for rect in tiny_plank:
            patch = patches.Rectangle(rect['xy'], rect['width'], rect['height'], angle=rect['angle'], linewidth=2, edgecolor='none', facecolor='#FF0000', transform=ax.transData)
            ax.add_patch(patch)

        # for rect in full_plank:
        #     patch = patches.Rectangle(rect['xy'], rect['width'], rect['height'], angle=rect['angle'], linewidth=2, edgecolor='none', facecolor='#E1F9F3', transform=ax.transData)
        #     ax.add_patch(patch)

        for ball in balls:
            patch = patches.Circle(ball['xy'], max(ball['radius'], 1), linewidth=2, edgecolor='k', facecolor='k')
            ax.add_patch(patch)

        ax.set_xlim(world_box[2], world_box[3])
        ax.set_ylim(world_box[0], world_box[1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # plt.axis('equal')

        filename = os.path.join(file_path, f'{base_filename}{i}.png')
        plt.savefig(filename)
        filenames.append(filename)

    plt.close(fig)

    i = 1
    while os.path.isfile(os.path.join(file_path, f"{file_name}{i}.gif")):
        i += 1
    next_filename = os.path.join(file_path, f"{file_name}{i}.gif")
    with imageio.get_writer(next_filename, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    for filename in filenames:
        os.remove(filename)