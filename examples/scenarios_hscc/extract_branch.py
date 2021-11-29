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

def find_edge_to_mode(edge_list, dest_idx):
    for idx in range(len(edge_list)):
        if edge_list[idx][1] == dest_idx:
            return idx
    print("Edge not found")
    return None

def find_edge(edge_list, src_idx, dest_idx):
    idx_list = find_edge_from_mode(edge_list, src_idx)
    for i in idx_list:
        if edge_list[i][1] == dest_idx:
            return i
    print(f"Edge ({src_idx}, {dest_idx}) not found")
    return None

def extract_branch(fn, height_high = 5, height_low = 0):
    config = {}
    with open(fn,'r') as f:
        config = json.load(f)

    agent = config["agents"][0]
    agent = split_modes(agent)
    edge_list = agent["edge_list"]
    guards = agent["guards"]
    mode_list = agent["mode_list"]
    time_horizon = agent["timeHorizons"]

    init_mode_id = agent["initialModeID"]
    traversal_queue = [(init_mode_id,0)]

    max_depth = -1
    max_depth_mode = init_mode_id
    
    while traversal_queue:
        tmp = traversal_queue.pop(0)
        mode_id = tmp[0]
        depth = tmp[1]
        if depth > max_depth:
            max_depth_mode = mode_id
            max_depth = depth
        
        edge_from_mode = find_edge_from_mode(edge_list, mode_id)
        for edge_id in edge_from_mode:
            dest_id = edge_list[edge_id][1]
            traversal_queue.append((dest_id,depth+1))

    print(max_depth, max_depth_mode)
    curr_mode = max_depth_mode
    longest_branch = [curr_mode]
    while curr_mode != init_mode_id:
        tmp_edge = find_edge_to_mode(edge_list, curr_mode)
        longest_branch.insert(0, edge_list[tmp_edge][0])
        curr_mode = edge_list[tmp_edge][0]

    new_mode_list = [mode_list[longest_branch[0]]]
    new_guard_list = []
    new_edge_list = []
    new_time_horizon = [time_horizon[longest_branch[0]]]
    for i in range(1,len(longest_branch)):
        new_mode_list.append(mode_list[longest_branch[i]])
        edge_idx = find_edge(edge_list, longest_branch[i-1], longest_branch[i])
        new_guard_list.append(guards[edge_idx])
        new_edge_list.append([i-1, i])
        new_time_horizon.append(time_horizon[longest_branch[i]])

    agent["edge_list"] = new_edge_list
    agent["guards"] = new_guard_list
    agent["timeHorizons"] = new_time_horizon
    agent["mode_list"] = new_mode_list

    config["agents"][0] = agent
    out_fn = fn.replace(".json","_TACAS.json")
    with open(out_fn,'w+') as f:
        json.dump(config,f)    

if __name__ == "__main__":
    fn = "./comp2D-1_quadNoNN_SymTR_10.json"
    extract_branch(fn)