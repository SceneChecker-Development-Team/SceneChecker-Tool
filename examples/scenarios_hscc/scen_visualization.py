import json
import numpy as np
import copy
import polytope as pc
import matplotlib.pyplot as plt
import pypoman as ppm

def split_modes(agent_info):
    seg_length = agent_info["segLength"]

    short_mode_list = []
    short_mode_dict = {}  # key: short mode parameter, value: short mode index
    long_short_mode_dict = {}  # key: original mode index, value: short mode index
    time_horizon_list = agent_info['timeHorizons']
    short_time_horizon_list = []
    # Create short modes
    for ind in range(len(agent_info["mode_list"])):
        mode = agent_info["mode_list"][ind]
        mode_params = np.array(mode[1]).astype(float)
        long_short_mode_dict[ind] = []
        if mode_params.shape[0] == 4:
            mode_params_src = mode_params[0:2]
            mode_params_dest = mode_params[2:]
        else:
            mode_params_src = mode_params[0:3]
            mode_params_dest = mode_params[3:]

        mode_length = np.linalg.norm(mode_params_dest - mode_params_src)
        mode_vector = seg_length * (mode_params_dest - mode_params_src) / np.linalg.norm(
            mode_params_dest - mode_params_src)
        tmp_mode_src = mode_params_src
        tmp_mode_dest = mode_params_src + mode_vector
        while np.linalg.norm(mode_params_dest - tmp_mode_src) >= seg_length:
            tmp_mode_params = list(tmp_mode_src) + list(tmp_mode_dest)
            short_mode_list.append(['follow_waypoint', tmp_mode_params])
            short_mode_dict[tuple(tmp_mode_params)] = len(short_mode_list) - 1
            long_short_mode_dict[ind].append(len(short_mode_list) - 1)
            short_time_horizon_list.append(round(seg_length * time_horizon_list[ind] / mode_length, 2))
            tmp_mode_src += mode_vector
            tmp_mode_dest += mode_vector
        if np.linalg.norm(mode_params_dest - tmp_mode_src) > 0.01:
            tmp_mode_params = list(tmp_mode_src) + list(mode_params_dest)
            short_mode_list.append(['follow_waypoint', tmp_mode_params])
            short_mode_dict[tuple(tmp_mode_params)] = len(short_mode_list) - 1
            long_short_mode_dict[ind].append(len(short_mode_list) - 1)
            short_time_horizon_list.append(
                round(np.linalg.norm(mode_params_dest - tmp_mode_src) * time_horizon_list[ind] / mode_length, 2))

    short_edge_list = []
    short_guard_list = []
    # Create more edges in edge lists
    for ind, edge in enumerate(agent_info['edge_list']):
        mode_src = edge[0]
        mode_dest = edge[1]
        # guard = np.array(agent_info['guards'][ind][1])
        new_edge = [long_short_mode_dict[mode_src][-1], long_short_mode_dict[mode_dest][0]]
        short_edge_list.append(new_edge)
        short_guard_list.append(agent_info['guards'][ind])

    guard = np.array(agent_info['guards'][ind][1])
    guard_radius = (guard[1, :] - guard[0, :]) / 2
    for ind in range(len(agent_info["mode_list"])):
        for j in range(1, len(long_short_mode_dict[ind])):
            short_edge_list.append([long_short_mode_dict[ind][j - 1], long_short_mode_dict[ind][j]])
            wp = short_mode_list[long_short_mode_dict[ind][j - 1]][1]
            if len(wp) == 4:
                wp = wp[2:]
                wp = np.array(wp + [0])
            else:
                wp = wp[3:]
                wp = np.array(wp + [0, 0, 0])

            guard = [list(wp - guard_radius), list(wp + guard_radius)]
            short_guard_list.append(['box', guard])

    orig_init_mode_id = agent_info['initialModeID']
    short_init_mode_id = long_short_mode_dict[orig_init_mode_id][0]

    tmp = copy.deepcopy(agent_info)
    tmp['timeHorizons'] = short_time_horizon_list
    tmp['guards'] = short_guard_list
    tmp['edge_list'] = short_edge_list
    tmp['mode_list'] = short_mode_list
    tmp['initialModeID'] = short_init_mode_id

    # with open('tmp.json','w+') as f:
    #     json.dump(tmp,f)
    return tmp

def find_edge_from_mode(edge_list, src_idx):
    idx_list = []
    for idx in range(len(edge_list)):
        if edge_list[idx][0] == src_idx:
            idx_list.append(idx)
    return idx_list

def ScenVisualization(fn):
    plt.figure()
    currentAxis = plt.gca()

    config = {}
    with open(fn,'r') as f:
        config = json.load(f)

    agent = config["agents"][0]
    edge_list = agent["edge_list"]
    guards = agent["guards"]
    mode_list = agent["mode_list"]
    unsafe_list = config["unsafeSet"]
    initialSet = agent["initialSet"]

    x_max = 0
    x_min = 10000
    y_max = 0
    y_min = 1000

    # plot modes
    init_mode_id = agent["initialModeID"]
    for i in range(1, len(mode_list)):
        mp = mode_list[i][1]
        if len(mp) == 4:
            plt.plot([mp[0],mp[2]], [mp[1],mp[3]], 'k', label='waypoints', linewidth= 0.7, markersize=5)
            plt.plot([mp[0],mp[2]], [mp[1],mp[3]], 'k.', label='waypoints', linewidth= 0.7, markersize=5)
        elif len(mp) == 6:
            plt.plot([mp[0],mp[3]], [mp[1],mp[4]], 'k', label='waypoints', linewidth= 0.7, markersize=5)
            plt.plot([mp[0],mp[3]], [mp[1],mp[4]], 'k.', label='waypoints', linewidth= 0.7, markersize=5)
    if init_mode_id == 0:
        mp = mode_list[0][1]
        if len(mp) == 4:
            plt.plot([mp[0],mp[2]], [mp[1],mp[3]], 'k', label='waypoints', linewidth= 0.7, markersize=5)
            plt.plot([mp[0],mp[2]], [mp[1],mp[3]], 'k.', label='waypoints', linewidth= 0.7, markersize=5)
        elif len(mp) == 6:
            plt.plot([mp[0],mp[3]], [mp[1],mp[4]], 'k', label='waypoints', linewidth= 0.7, markersize=5)
            plt.plot([mp[0],mp[3]], [mp[1],mp[4]], 'k.', label='waypoints', linewidth= 0.7, markersize=5)
    
    color = '#d9d9d9'
    for unsafe in unsafe_list:
        if unsafe[0] == "Vertices":
            unsafe_poly = pc.qhull(np.array(unsafe[1]))
        elif unsafe[0] == "Matrix":
            unsafe_poly = pc.Polytope(np.array(unsafe[1][0]),np.array(unsafe[1][1]))
        else:
            unsafe_poly = pc.box2poly(np.array(unsafe[1]).T)    
        poly = pc.projection(unsafe_poly, [1, 2])
        if poly.A.size != 0:
            vert = ppm.duality.compute_polytope_vertices(poly.A, poly.b)
            ppm.polygon.plot_polygon(vert, color=color, alpha = 1)
            vert = np.array(vert)
            x = vert[:, 0]
            x_max = max(x_max, np.max(x))
            x_min = min(x_min, np.min(x))
            y = vert[:, 1]
            y_max = max(y_max, np.max(y))
            y_min = min(y_min, np.min(y))

    color = '#80b1d3'
    poly = pc.box2poly(np.array(initialSet[1]).T)
    poly = pc.projection(poly, [1, 2])
    vert = ppm.duality.compute_polytope_vertices(poly.A, poly.b)
    ppm.polygon.plot_polygon(vert, color=color, alpha = 1)

    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('x', fontsize=15)
    plt.ylabel('y', fontsize=15)
    plt.xlim([x_min-10,x_max+10])
    plt.ylim([y_min-10,y_max+10])

    plt.show()


if __name__ == "__main__":
    fn = "./simp2D-2_carNoNN_SymTR_10.json"
    ScenVisualization(fn)
