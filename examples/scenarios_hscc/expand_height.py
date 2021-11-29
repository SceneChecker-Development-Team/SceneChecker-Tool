import json
import numpy as np
import copy

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

def expand_height(fn, height_high = 5, height_low = 0):
    config = {}
    with open(fn,'r') as f:
        config = json.load(f)

    agent = config["agents"][0]
    agent = split_modes(agent)
    edge_list = agent["edge_list"]
    guards = agent["guards"]
    mode_list = agent["mode_list"]

    init_mode_id = agent["initialModeID"]
    initial_mode_parameters = mode_list[init_mode_id][1]
    initial_mode_parameters[2] = height_low
    initial_mode_parameters[5] = height_high
    mode_list[init_mode_id][1] = initial_mode_parameters

    traversal_queue = [init_mode_id]
    while traversal_queue:
        mode_id = traversal_queue.pop(0)
        src_mode_parameters = mode_list[mode_id][1]
        edge_from_mode = find_edge_from_mode(edge_list, mode_id)
        for edge_id in edge_from_mode:
            edge = edge_list[edge_id]
            dest_id = edge[1]
            dest_mode_parameters = mode_list[dest_id][1]
            dest_mode_parameters[2] = src_mode_parameters[5]
            if dest_mode_parameters[2] == height_high:
                dest_mode_parameters[5] = height_low
            else:
                dest_mode_parameters[5] = height_high
            mode_list[dest_id][1] = dest_mode_parameters

            if dest_mode_parameters[2] == height_high:
                guards[edge_id][1][0][2] = height_high - 0.5
                guards[edge_id][1][1][2] = height_high + 0.5
            else:
                guards[edge_id][1][0][2] = height_low - 0.5
                guards[edge_id][1][1][2] = height_low + 0.5

            traversal_queue.append(dest_id)

    config["agents"][0] = agent
    out_fn = fn.replace("2D","3D")
    with open(out_fn,'w') as f:
        json.dump(config,f)    

if __name__ == "__main__":
    fn = "./comp2D-1_quadNoNN_SymTR_10_flowstar.json"
    expand_height(fn)