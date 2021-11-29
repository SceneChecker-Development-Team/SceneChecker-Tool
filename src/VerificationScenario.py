import json
import os
from src.Waypoint import Waypoint
from src.Edge import Edge
from src.Unsafeset import Unsafeset
from src.Agent import Agent, TubeMaster, DRONE_TYPE
from src.Plot import Plotter
# from src.Plot_3d import Plotter
from src.ReachtubeSegment import ReachtubeSegment
import numpy as np
from typing import List, Dict, Optional, Tuple
import polytope as pc
from src.PolyUtils import PolyUtils
import math
import cProfile
import time
import pdb
from src.ParseModel import importFunctions
import subprocess
from src.dryvr_parser import dryvr_parse
# from mayavi import mlab
import os
import sys
import copy
from collections import deque
import importlib 
import matplotlib.pyplot as plt
import sys
import datetime

class VerificationScenario:

    def __init__(self, json_input_file: str, json_output_file: str = None, log_file = None, seed = True) -> None:
        self.json_input_file = json_input_file
        self.log_file = log_file
        self.old_stdout = sys.stdout
        if log_file is not None:
            sys.stdout = self.log_file
        
        try:
            print(">>>>>>",json_input_file)
            with open(json_input_file) as json_input:
                json_result = json.load(json_input)
        except FileNotFoundError:
            print("Json Scenario File", json_input_file, "not found")

        self.json_output_file = json_output_file
        agents_info_list = json_result["agents"]
        self.num_agents: int = len(agents_info_list)
        self.time_step: float = json_result["time_step"]
        self.reachability_engine: str = json_result["reachability_engine"]
        # TODO make these agent ID's Dynamically mapped with a hash of the dynamics summary
        self.agent_dynamics_ids: List[str] = [agent_info["directory"] for agent_info in agents_info_list]
        # list of polytopes one for each agent.
        initset_list = [agent_info["initialSet"][1] for agent_info in agents_info_list]
        self.agents_variables_list = [agent_info["variables"] for agent_info in agents_info_list]
        self.initsets: List[pc.Polytope] = [pc.box2poly(np.array(initset).T) for initset in initset_list]
        if "goalSet" in json_result:
            self.goalsets: List[pc.Polytope] = [pc.box2poly(np.array(goalset[1]).T) for goalset in json_result["goalSet"]]
        else:
            self.goalsets = []
        # TODO add support for controller level grid resolutions
        self.grid_resolution: np.array = np.array(json_result["grid_resolution"])
        # TODO Add waypoint IO

        self.tmp = False
        if "qnn_s4" in json_input_file:
            self.tmp = True

        self.safe_plot = False
        if "unSafe" in json_input_file:
            self.safe_plot = True

        self.agents_list: List[Agent]
        if "symmetry_level" in json_result and json_result["symmetry_level"].lower() == "0":
            self.sym_level = 0
            print("Warning, not using symmetry")
        elif "symmetry_level" in json_result and json_result["symmetry_level"].lower() == "1":
            self.sym_level = 1
            print("Warning, symmetry is used WITHOUT fixed point checking")
        else:
            self.sym_level = 2
            print("Great, symmetry is used WITH fixed point checking")

        # TODO add support for Unsafe Set Class
        # self.global_unsafe_set: List[pc.Polytope] = [
        #     Unsafeset(i, pc.box2poly(np.array(json_result["unsafeSet"][i][1]).T)) for i in
        #     range(len(json_result["unsafeSet"]))]
        self.global_unsafe_set = []
        for i in range(len(json_result["unsafeSet"])):
            unsafe_set = json_result["unsafeSet"][i]
            if unsafe_set[0] == "Box":
                self.global_unsafe_set.append(Unsafeset(i,pc.box2poly(np.array(unsafe_set[1]).T)))

            elif unsafe_set[0] == "Vertices":
                self.global_unsafe_set.append(Unsafeset(i,pc.qhull(np.array(unsafe_set[1]))))
            elif unsafe_set[0] == "Matrix":
                self.global_unsafe_set.append(Unsafeset(i, pc.Polytope(np.array(unsafe_set[1][0]),np.array(unsafe_set[1][1]))))
            else:
                raise Exception("Unsupported unsafe type", unsafe_set[0])


        self.agents_list = VerificationScenario.get_agent_list_from_info(agents_info_list, self.global_unsafe_set,
                                                                         snowflake_path=False)
        if seed:
            # if "cnn_s4" in json_input_file:
            #     self.agents_list[0].seed = 4
            # elif "qnn_s3T_dr" in json_input_file:
            #     self.agents_list[0].seed = 4
            # else:
            #     self.agents_list[0].seed = 4
            self.agents_list[0].seed = 4
        else:
            self.agents_list[0].seed = None

        self.refine_threshold = json_result["refine_threshold"]

        self.max_dryvr_time_horizon = 10000

        if not json_output_file is None:
            self.reset_to_vir_reset_str_fn = importFunctions(agents_info_list[0]["directory"])[
                8]  # TODO: change when more than one agent

        # dryvr_plotter_command = "python " + dryvr_path + "retrieve_dryvr_tube.py " + dryvr_path + "output/reachtube.txt"

        # self.json_output.write
        # self.json_output.write(str(json.dumps({"vertex": dryvr_vertex_list,
        #                                       "edge": dryvr_edge_list,
        #                                       "variables": dryvr_variables_list,
        #                                       "guards": dryvr_guards_list,
        #                                       "resets": dryvr_resets_list,
        #                                       "initialSet": dryvr_initset_list,
        #                                       "unsafeSet": dryvr_unsafeset_string,
        #                                       "timeHorizon": dryvr_timehorizon,
        #                                       "directory": dryvr_directory_string})))

        self.tube_master: TubeMaster = TubeMaster(self.grid_resolution, {key: self.reachability_engine for key
                                                                         in set(self.agent_dynamics_ids)},
                                                  self.agents_list, self.sym_level)

        self.agents_full_tubes: List[List[ReachtubeSegment]] = []
        self.agents_full_trace: List[List[np.array]] = []
        self.reached_fixed_point = False
        # self.f = open(json_input_file[:-5] + ".txt", "w+")

    def construct_abs_dryvr_input_file(self, dryvr_time_bound):
        dryvr_vertex_list = []
        dryvr_edge_list = []
        dryvr_guards_list = []
        dryvr_resets_list = []
        dryvr_unsafeset_string = ""
        dryvr_directory_string = "examples/Linear3D"  # self.dryvr_path +
        dryvr_mode_to_id_dict = {}
        dryvr_mode_counter = 0

        for virtual_mode in self.abs_initset:
            dryvr_mode = str(virtual_mode)
            dryvr_mode = dryvr_mode.replace(",", ";")
            self.dryvrmode_to_virtualmode[dryvr_mode + "," + str(dryvr_mode_counter)] = virtual_mode
            dryvr_vertex_list.append(dryvr_mode)
            dryvr_mode_to_id_dict[virtual_mode] = dryvr_mode_counter
            dryvr_mode_counter += 1
        dryvr_variables_list = self.agents_variables_list[0]  ## change when we have more than one agent

        for virtual_edge in self.abs_edges_guards:
            first_mode = virtual_edge[0]
            second_mode = virtual_edge[1]
            first_mode_id = dryvr_mode_to_id_dict[first_mode]
            second_mode_id = dryvr_mode_to_id_dict[second_mode]
            dryvr_edge_list.append([first_mode_id, second_mode_id])
            guard_rect = PolyUtils.get_region_bounding_box(self.abs_edges_guards_union[virtual_edge])
            guard_string = "And("
            for i in range(len(dryvr_variables_list)):
                guard_string += "-1 * " + dryvr_variables_list[i] + "<=" + str(-1 * guard_rect[0][i]) + "," \
                                + dryvr_variables_list[i] + "<=" + str(guard_rect[1][i]) + ","
            guard_string += "t<=" + str(self.abs_timebound[virtual_edge]) + ",-t<=0.0"
            guard_string = guard_string + ")"
            state_dim = len(dryvr_variables_list)
            print("resets transform information: ", self.abs_edges_guards[virtual_edge][0][1],
                  self.abs_edges_guards[virtual_edge][0][2])
            reset_string = self.reset_to_vir_reset_str_fn(np.array([0] * state_dim),
                                                          # TODO: remove this as it's not used
                                                          self.abs_edges_guards[virtual_edge][0][1],
                                                          # get the first element of the guard list transform information
                                                          self.abs_edges_guards[virtual_edge][0][2])
            dryvr_resets_list.append(reset_string)
            dryvr_guards_list.append(guard_string)

        dryvr_initset_list = self.abs_agent_initset[-1].tolist()
        dryvr_timehorizon = 0  # TODO: should be changed
        if self.max_dryvr_time_horizon == 10000:
            for virtual_edge in self.abs_timebound:
                dryvr_timehorizon = dryvr_timehorizon + self.abs_timebound[virtual_edge]
            self.max_dryvr_time_horizon = dryvr_timehorizon

        if dryvr_time_bound < self.max_dryvr_time_horizon:
            dryvr_timehorizon = dryvr_time_bound
        else:
            dryvr_timehorizon = self.max_dryvr_time_horizon

        # self.dryvr_path + "input/nondaginput/" +
        try:
            self.json_output = open(self.dryvr_path + "input/nondaginput/" + self.json_output_file, "w")
        except IOError:
            print('File does not exist')

        json.dump({"vertex": dryvr_vertex_list,
                   "edge": dryvr_edge_list,
                   "variables": dryvr_variables_list,
                   "guards": dryvr_guards_list,
                   "resets": dryvr_resets_list,
                   "initialSet": dryvr_initset_list,
                   "unsafeSet": dryvr_unsafeset_string,
                   "timeHorizon": dryvr_timehorizon,
                   "directory": dryvr_directory_string,
                   "initialVertex": 0,
                   "seed": 664891646,
                   "bloatingMethod": "PW",
                   "kvalue": [1] * len(dryvr_variables_list),
                   }, self.json_output, indent=2)
        self.json_output.close()

    def run_dryvr(self):
        cur_folder_path = os.getcwd()
        os.chdir(self.dryvr_path)
        # dryvr_call_command = "python " + "main.py " + "input/nondaginput/" + self.json_output_file
        # print("dryvr_call_command: ", dryvr_call_command)
        # process = subprocess.Popen(dryvr_call_command.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()
        params = ["python", "main.py", "input/nondaginput/" + self.json_output_file]
        # params = ["./dryvr_call_script.sh"]
        # params = ["pwd"]

        result = str(subprocess.check_output(params)).split('\\n')
        os.chdir(cur_folder_path)
        print("dryvr output: ", result)

    def get_dryvr_tube(self, node):
        #	fig1 = plt.figure()
        #	ax1 = fig1.add_subplot('111')
        lowerBound = []
        upperBound = []
        tube = []
        for key in sorted(node.lowerBound):
            lowerBound.append(node.lowerBound[key])
        for key in sorted(node.upperBound):
            upperBound.append(node.upperBound[key])

        for i in range(min(len(lowerBound), len(upperBound))):
            lb = list(map(float, lowerBound[i]))
            lb.pop(0)
            ub = list(map(float, upperBound[i]))
            ub.pop(0)  # TODO: removing time for now
            if len(lb) != len(self.agents_variables_list[0]) or len(ub) != len(self.agents_variables_list[0]):
                continue
            tube.append(np.array([lb, ub]))
            # tube.append(ub)

        return tube

    def parse_dryvr_output(self):
        try:
            file = open(self.dryvr_path + "output/reachtube.txt", 'r')
        except IOError:
            print('File does not exist')
        lines = file.readlines()
        initNode, y_min, y_max = dryvr_parse(lines)

        # Using DFS algorithm to Draw image per Node
        stack = [initNode]
        while stack:
            curNode = stack.pop()
            for c in curNode.child:
                stack.append(curNode.child[c])
            tube = self.get_dryvr_tube(curNode)
            print("curNode: ", self.abs_reach[self.dryvrmode_to_virtualmode[curNode.nodeId]])
            print("curNode tube: ", tube)
            self.abs_reach[self.dryvrmode_to_virtualmode[curNode.nodeId]] = PolyUtils.merge_tubes([tube,
                                                                                                   self.abs_reach[
                                                                                                       self.dryvrmode_to_virtualmode[
                                                                                                           curNode.nodeId]]])
            print("the tube of parsed mode ", curNode.nodeId, ", denoted by ",
                  self.dryvrmode_to_virtualmode[curNode.nodeId], "is: ",
                  self.abs_reach[self.dryvrmode_to_virtualmode[curNode.nodeId]])

    @staticmethod
    def split_modes(agent_info):
        seg_length = agent_info["segLength"]

        short_mode_list = []
        short_mode_dict = {} # key: short mode parameter, value: short mode index
        long_short_mode_dict = {} # key: original mode index, value: short mode index
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
            mode_vector = seg_length * (mode_params_dest - mode_params_src) / np.linalg.norm(mode_params_dest - mode_params_src)
            tmp_mode_src = mode_params_src
            tmp_mode_dest = mode_params_src + mode_vector
            while np.linalg.norm(mode_params_dest - tmp_mode_src) >= seg_length:
                tmp_mode_params = list(tmp_mode_src)+list(tmp_mode_dest)
                short_mode_list.append(['follow_waypoint',tmp_mode_params])
                short_mode_dict[tuple(tmp_mode_params)] = len(short_mode_list) - 1
                long_short_mode_dict[ind].append(len(short_mode_list) - 1)
                short_time_horizon_list.append(round(seg_length * time_horizon_list[ind] / mode_length,2))
                tmp_mode_src += mode_vector
                tmp_mode_dest += mode_vector
            if np.linalg.norm(mode_params_dest - tmp_mode_src) > 0.01:
                tmp_mode_params = list(tmp_mode_src) + list(mode_params_dest)
                short_mode_list.append(['follow_waypoint', tmp_mode_params])
                short_mode_dict[tuple(tmp_mode_params)] = len(short_mode_list) - 1
                long_short_mode_dict[ind].append(len(short_mode_list) - 1)
                short_time_horizon_list.append(round(np.linalg.norm(mode_params_dest - tmp_mode_src) * time_horizon_list[ind] / mode_length,2))

        short_edge_list = []
        short_guard_list = []
        # Create more edges in edge lists
        for ind, edge in enumerate(agent_info['edge_list']):
            mode_src = edge[0]
            mode_dest = edge[1]
            # guard = np.array(agent_info['guards'][ind][1])
            new_edge = [long_short_mode_dict[mode_src][-1],long_short_mode_dict[mode_dest][0]]
            short_edge_list.append(new_edge)
            short_guard_list.append(agent_info['guards'][ind])

        guard = np.array(agent_info['guards'][ind][1])
        guard_radius = (guard[1,:] - guard[0,:])/2
        for ind in range(len(agent_info["mode_list"])):
            for j in range(1,len(long_short_mode_dict[ind])):
                short_edge_list.append([long_short_mode_dict[ind][j-1],long_short_mode_dict[ind][j]])
                wp = short_mode_list[long_short_mode_dict[ind][j-1]][1]
                if len(wp) == 4:
                    wp = wp[2:]
                    wp = np.array(wp + [0])
                else:
                    wp = wp[3:]
                    wp = np.array(wp + [0,0,0])

                guard = [list(wp-guard_radius),list(wp+guard_radius)]
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

    @staticmethod
    def get_agent_list_from_info(agents_info_list, unsafeset_list=None, snowflake_path=False):
        unsafeset_list = copy.deepcopy(unsafeset_list)
        agents_list = []
        if snowflake_path:
            P = VerificationScenario.generate_snowflake_waypoints(80, 2)
            rows_to_be_deleted = []
            for i in range(P.shape[0] - 1):
                if np.count_nonzero(P[i, :]) == 0 and np.count_nonzero(P[i + 1, :]) == 0:
                    rows_to_be_deleted.append(i)
            P = np.delete(P, rows_to_be_deleted, 0)
            P = np.hstack(
                (P, np.zeros((P.shape[0], 1))))  # HUSSEIN: UNCOMMENT IF 3D LINEAR EXAMPLE, COMMENT IF FIXEDWING

        for agent_info in agents_info_list:
            mode_list: List[Waypoint] = []
            edge_list: List[Edge] = []
            mode_neighbors: Dict[int, List[int]] = {}
            mode_parents: Dict[int, List[int]] = {}
            initial_set: pc.Polytope
            dynamics: str
            abs_mode_list: List[Waypoint] = []
            abs_edge_list: List[Edge] = []
            abs_mode_neighbors: Dict[int, List[int]] = {}
            abs_mode_parents: Dict[int, List[int]] = {}
            abs_initial_set: pc.Region
            function_arr = importFunctions(agent_info["directory"])
            get_transform_information = function_arr[1]
            transform_poly_to_virtual = function_arr[2]
            transform_mode_to_virtual = function_arr[3]

            mode_to_abs_mode: Dict[int, int] = {}
            abs_mode_to_mode: Dict[int, List[int]] = {}

            initial_set = pc.box2poly(np.array(agent_info["initialSet"][1]).T)
            print(len(unsafeset_list))

            if 'segLength' in agent_info and agent_info['segLength'] > 0:
                agent_info = VerificationScenario.split_modes(agent_info)
            guard_list = agent_info["guards"]

            for ind, mode in enumerate(agent_info["mode_list"]):
                try:
                    unsafeset_mode = []
                    for unsafeset in unsafeset_list:
                        if len(mode[1]) == 6:
                            unsafe_box = PolyUtils.get_bounding_box(unsafeset.unsafe_set)[:,0:3]
                            mode_parameter1 = np.array(mode[1])[0:3]
                            mode_parameter2 = np.array(mode[1])[3:]

                        else:
                            unsafe_box = PolyUtils.get_bounding_box(unsafeset.unsafe_set)[:,0:2]
                            mode_parameter1 = np.array(mode[1])[0:2]
                            mode_parameter2 = np.array(mode[1])[2:]
                        seg_length = np.linalg.norm(mode_parameter2 - mode_parameter1)
                        if PolyUtils.dist_to_polytope(unsafe_box,mode_parameter1) < 5 * seg_length or PolyUtils.dist_to_polytope(unsafe_box,mode_parameter2) < 5 * seg_length:
                            # print(mode[1])
                            # print(PolyUtils.get_bounding_box(unsafeset.unsafe_set))
                            # print(PolyUtils.dist_to_polytope(unsafe_box,mode_parameter1),PolyUtils.dist_to_polytope(unsafe_box,mode_parameter2))
                            unsafeset_mode.append(unsafeset)
                    # print(len(unsafeset_mode))
                    mode_list.append(
                        Waypoint(mode[0], mode[1], agent_info["timeHorizons"][ind], ind, unsafeset_list=unsafeset_mode))
                    mode_neighbors[mode_list[ind].id] = []
                    mode_parents[mode_list[ind].id] = []
                    transform_information = get_transform_information(mode_list[ind])

                    if ind == agent_info['initialModeID']:
                        abs_initial_set = transform_poly_to_virtual(initial_set, transform_information)
                        # abs_agent_initaset.append(PolyUtils.get_bounding_box(agents_list[-1].transform_poly_to_virtual(
                        #    agents_list[-1].initial_set, transform_information1)))

                    abs_mode = transform_mode_to_virtual(mode_list[ind], transform_information)
                    abs_unsafeset_list = []
                    for unsafeset in unsafeset_mode:
                        abs_unsafeset_list.append(Unsafeset(ind,
                                                            transform_poly_to_virtual(unsafeset.unsafe_set,
                                                                                      transform_information)))

                    abs_mode_exists = False
                    for i in range(len(abs_mode_list)):
                        other_abs_mode = abs_mode_list[i]
                        if abs_mode.is_equal(other_abs_mode):
                            abs_mode_exists = True
                            abs_mode_to_mode[other_abs_mode.id].append(ind)
                            abs_mode_list[i].time_bound = max(abs_mode_list[i].time_bound, mode_list[ind].time_bound)
                            abs_mode_list[i].unsafeset_list.extend(abs_unsafeset_list)
                            mode_to_abs_mode[mode_list[ind].id] = i
                            break
                    if not abs_mode_exists:
                        abs_mode.id = len(abs_mode_list)
                        abs_mode.unsafeset_list = abs_unsafeset_list
                        abs_mode_list.append(abs_mode)
                        abs_mode_neighbors[abs_mode.id] = []
                        abs_mode_parents[abs_mode.id] = []
                        abs_mode_to_mode[abs_mode.id] = [mode_list[ind].id]
                        mode_to_abs_mode[mode_list[ind].id] = abs_mode.id

                except IndexError:
                    pdb.set_trace()

            for ind, edge in enumerate(agent_info["edge_list"]):
                try:
                    edge_list.append(
                        Edge(edge[0], edge[1], ind,
                             np.array(guard_list[ind][1])))
                    edge = edge_list[ind]
                    if edge.source in mode_neighbors:
                        mode_neighbors[edge.source].append(ind)
                    if edge.dest in mode_parents:
                        mode_parents[edge.dest].append(ind)

                    transform_information_src = get_transform_information(mode_list[edge.source])
                    transform_information_dest = get_transform_information(mode_list[edge.dest])
                    abs_guard = transform_poly_to_virtual(
                        pc.box2poly(np.array(edge.guard).T),
                        transform_information_src)
                    abs_mode_id_src = mode_to_abs_mode[edge.source]
                    abs_mode_id_dest = mode_to_abs_mode[edge.dest]

                    abs_edge = Edge(abs_mode_id_src, abs_mode_id_dest, 0, [])
                    abs_edge_exists = False
                    for i in range(len(abs_edge_list)):
                        other_abs_edge = abs_edge_list[i]
                        if abs_edge.is_equal(other_abs_edge):
                            abs_edge_exists = True
                            abs_edge_list[i].guard.append((PolyUtils.get_bounding_box(abs_guard),
                                                           transform_information_src,
                                                           transform_information_dest, ind))
                            abs_edge_list[i].region_guard = PolyUtils.get_region_union(other_abs_edge.region_guard,
                                                                                       abs_guard)
                            break
                    if not abs_edge_exists:
                        abs_edge.id = len(abs_edge_list)
                        abs_edge.guard = [(PolyUtils.get_bounding_box(abs_guard),
                                           transform_information_src,
                                           transform_information_dest, ind)]
                        abs_edge.region_guard = pc.Region(list_poly=[abs_guard])
                        abs_edge_list.append(abs_edge)
                        abs_mode_neighbors[abs_mode_id_src].append(abs_edge.id)
                        abs_mode_parents[abs_mode_id_dest].append(abs_edge.id)


                except IndexError:
                    pdb.set_trace()

            for abs_edge in abs_edge_list:
                if pc.is_empty(abs_edge.region_guard):
                    pdb.set_trace()
                for edge_info in abs_edge.guard:
                    if not edge_list[edge_info[3]].source in abs_mode_to_mode[abs_edge.source]:
                        pdb.set_trace()
                    if not edge_list[edge_info[3]].dest in abs_mode_to_mode[abs_edge.dest]:
                        pdb.set_trace()

            agents_list.append(
                Agent(agent_info["variables"], mode_list, edge_list,
                      mode_neighbors,
                      mode_parents,
                      initial_set, agent_info["initialModeID"], agent_info["directory"], function_arr, abs_mode_list,
                      abs_edge_list,
                      abs_mode_neighbors,
                      abs_mode_parents,
                      abs_initial_set, mode_to_abs_mode,
                      abs_mode_to_mode))

        return agents_list

    @staticmethod
    def generate_snowflake_waypoints(side_len=1, n=1):
        P = np.array([np.array([0, 0]), np.array([side_len, 0]),
                      np.array([side_len * math.cos(-math.pi / 3), side_len * math.sin(-math.pi / 3)]),
                      np.array([0, 0])])
        for i in range(n):
            newP = np.zeros((P.shape[0] * 4 + 1, 2))
            for j in range(P.shape[0] - 1):
                newP[4 * j + 1, :] = P[j, :]
                # print("newp: ", newP[4 * j + 1, :], "P: ", P[j, :])
                newP[4 * j + 2, :] = (2 * P[j, :] + P[j + 1, :]) / 3
                link = P[j + 1, :] - P[j, :]
                ang = math.atan2(link[1], link[0])
                linkLeng = math.sqrt(link[0] * link[0] + link[1] * link[1])
                newP[4 * j + 3, :] = newP[4 * j + 2, :] + (linkLeng / 3) * np.array(
                    [math.cos(ang + math.pi / 3), math.sin(ang + math.pi / 3)])
                newP[4 * j + 4, :] = (P[j, :] + 2 * P[j + 1, :]) / 3

            newP[4 * P.shape[0], :] = P[P.shape[0] - 1, :]
            P = newP

        return P

    @staticmethod
    def get_agent_list_from_info_snowflake(agents_info_list):
        agents_list = []
        abs_edges_guards: Dict[Tuple[Tuple, ...], List[Tuple[np.array, Tuple[float, ...], Tuple[float, ...]]]] = {}
        abs_reach: Dict[Tuple[float, ...], pc.Region] = {}
        abs_initset: Dict[Tuple[List[float]], pc.Region] = {}
        abs_node_neighbors: Dict[Tuple[List[float]], List[Tuple[List[float]]]] = {}
        P = VerificationScenario.generate_snowflake_waypoints(80, 2)
        rows_to_be_deleted = []
        for i in range(P.shape[0] - 1):
            if np.count_nonzero(P[i, :]) == 0 and np.count_nonzero(P[i + 1, :]) == 0:
                rows_to_be_deleted.append(i)
        P = np.delete(P, rows_to_be_deleted, 0)
        P = np.hstack((P, np.zeros((P.shape[0], 1))))  # HUSSEIN: UNCOMMENT IF 3D LINEAR EXAMPLE, COMMENT IF FIXEDWING
        abs_agent_initset = []
        abs_timebound = {}
        for agent_info in agents_info_list:
            waypoint_list = []
            for ind in range(int(P.shape[0] / 3)):
                try:
                    # HUSSEIN: UNCOMMENT IF 3D LINEAR EXAMPLE, COMMENT IF FIXEDWING

                    waypoint_list.append(
                        Waypoint("follow_waypoint", [P[ind, 0], P[ind, 1], P[ind, 2]],
                                 [[P[ind, 0] - 1, P[ind, 1] - 1, P[ind, 2] - 100], [P[ind, 0]
                                                                                    + 1,
                                                                                    P[ind,
                                                                                      1] + 1,
                                                                                    P[ind,
                                                                                      2] + 100]],
                                 20))
                    """
                    waypoint_list.append(
                        Waypoint("follow_waypoint", [P[ind, 0], P[ind, 1]],
                                 [[P[ind, 0] - 4, P[ind, 1] - 4, -5, -1000], [P[ind, 0] + 4, P[ind, 1] + 4, 5, 1000]], 15))
                    """
                except IndexError:
                    pdb.set_trace()

            agents_list.append(Agent(waypoint_list, pc.box2poly(np.array(agent_info["initialSet"][1]).T),
                                     agent_info["directory"], importFunctions(agent_info["directory"])))
            for i in range(len(waypoint_list) - 1):
                waypoint = waypoint_list[i]
                if i == 0:
                    prev_waypoint = Waypoint("follow_waypoint",
                                             # np.average(PolyUtils.get_region_bounding_box(agents_list[-1].initial_set),
                                             #          axis=0),
                                             np.concatenate((np.average(
                                                 PolyUtils.get_region_bounding_box(agents_list[-1].initial_set),
                                                 axis=0)[0:3],
                                                             np.average(PolyUtils.get_region_bounding_box(
                                                                 agents_list[-1].initial_set),
                                                                 axis=0)[0:3])),
                                             PolyUtils.get_region_bounding_box(agents_list[-1].initial_set), 0)
                    transform_information1 = agents_list[-1].get_transform_information(waypoint.mode_parameters,
                                                                                       prev_waypoint.mode_parameters,
                                                                                       time_bound=waypoint.time_bound)
                    abs_agent_initset.append(agents_list[-1].transform_poly_to_virtual(
                        agents_list[-1].initial_set, transform_information1))

                    # print("initial set guard: ", PolyUtils.get_region_bounding_box(agents_list[-1].initial_set))
                else:
                    prev_waypoint = waypoint_list[i - 1]
                next_waypoint = waypoint_list[i + 1]
                transform_information1 = agents_list[-1].get_transform_information(waypoint.mode_parameters,
                                                                                   prev_waypoint.mode_parameters,
                                                                                   time_bound=waypoint.time_bound)
                transform_information2 = agents_list[-1].get_transform_information(next_waypoint.mode_parameters,
                                                                                   waypoint.mode_parameters,
                                                                                   time_bound=waypoint.time_bound)
                abs_prev_waypoint = agents_list[-1].transform_mode_to_virtual(prev_waypoint.mode_parameters,
                                                                              transform_information1)
                abs_waypoint = agents_list[-1].transform_mode_to_virtual(waypoint.mode_parameters,
                                                                         transform_information2)
                abs_node = tuple(abs_prev_waypoint)
                abs_edge = tuple([abs_node, tuple(abs_waypoint)])
                if abs_node in abs_node_neighbors:
                    abs_node_neighbors[abs_node].append(tuple(abs_waypoint))
                else:
                    abs_node_neighbors[abs_node] = [tuple(abs_waypoint)]
                if abs_edge in abs_edges_guards:
                    abs_edges_guards[abs_edge].append(
                        (PolyUtils.get_bounding_box(agents_list[-1].transform_poly_to_virtual(
                            pc.box2poly(waypoint.original_guard.T),
                            transform_information1)),
                         transform_information1,
                         transform_information2))
                    abs_timebound[abs_edge] = max(abs_timebound[abs_edge], waypoint.time_bound)
                else:
                    abs_edges_guards[abs_edge] = [(PolyUtils.get_bounding_box(agents_list[-1].transform_poly_to_virtual(
                        pc.box2poly(waypoint.original_guard.T), transform_information1)), transform_information1,
                                                   transform_information2)]
                    abs_timebound[abs_edge] = waypoint.time_bound

                if not abs_node in abs_reach:
                    abs_reach[abs_node] = pc.Region(list_poly=[])

                if not abs_node in abs_initset:
                    abs_initset[abs_node] = pc.Region(list_poly=[])

        return agents_list, abs_agent_initset, abs_timebound, abs_edges_guards, abs_reach, abs_initset, abs_node_neighbors

    @staticmethod
    def get_segment_unsafeset_inter_list(tubesegment: ReachtubeSegment, unsafeset_list: List[Unsafeset] = []) -> bool:
        if len(unsafeset_list) <= 0:
            print("unsafeset list empty")
            return []
        poly: pc.Polytope
        intersection_list = []
        for unsafeset in unsafeset_list:
            if unsafeset.mode_id in intersection_list:
                continue
            intersecting = False
            if type(unsafeset.unsafe_set) == pc.Polytope:
                unsafeset.unsafe_set = pc.Region(list_poly=[unsafeset.unsafe_set])
            for unsafe_poly in unsafeset.unsafe_set.list_poly:
                for tube in tubesegment.tube_list:
                    for reg in tube:
                        if type(reg) == pc.Polytope:
                            reg = pc.Region(list_poly=[reg])
                            for poly in reg.list_poly:
                                # TODO: Maybe don't project down before intersecting...
                                # we need the verify input format for this, but basically,
                                # we should let the user leave dimensions that don't matter
                                # as unbounded dimensions in the unsafe set.
                                # PolyUtils.print_region(region)
                                # unsafeset = PolyUtils.project_to_intersect(unsafeset, region)
                                # print("unsafeset:", unsafeset)
                                if not PolyUtils.is_polytope_intersection_empty(unsafe_poly, poly):
                                    intersection_list.append(unsafeset.mode_id)
                                    intersecting = True
                                    return intersection_list
                            if intersecting:
                                break
                    if intersecting:
                        break
                if intersecting:
                    break
        return intersection_list

    def compute_agent_full_tube(self, cur_agent: Agent, t = 0) -> Tuple[List[ReachtubeSegment], List[List[int]]]:
        full_tube: List[ReachtubeSegment] = []
        look_back: List[List[int]] = [[0, 0]]
        # list of pairs of index of a jump, look back number of indices
        # all polytopes in the tube between two consecutive indices would have a lookback equal to the previous one
        time_passed = 0
        visited = {}
        traversal_parent = {}
        # TODO update this loop to use the transform class
        #################################
        if self.sym_level != 2:
            root_initset: pc.Region = cur_agent.initial_set
            cur_initset = root_initset
            traversal_queue = deque([tuple([cur_initset, cur_agent.initial_mode_id, 0])])
            per_mode_initset = [np.array([])] * len(cur_agent.mode_list)
            # visited = [-1] * len(cur_agent.mode_list)
            # visited[cur_agent.initial_mode_id] = 1
            per_mode_initset[cur_agent.initial_mode_id] = cur_agent.initial_set
        else:
            transform_information = cur_agent.get_transform_information(cur_agent.mode_list[cur_agent.initial_mode_id])
            root_initset: pc.Region = cur_agent.transform_poly_to_virtual(cur_agent.initial_set, transform_information)
            cur_initset: pc.Region = root_initset
            traversal_queue = deque([tuple([cur_initset, cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id], 0])])
            refine_counter = 0
            traversal_parent[(cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id], 0)] = None
            # per_mode_initset = [np.array([])] * len(cur_agent.abs_mode_list)
            # visited = [-1] * len(cur_agent.abs_mode_list)
            # visited[cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id]] = 1
            # per_mode_initset[cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id]] = cur_initset
        # \
        #                 and cur_agent.number_of_segments_computed < len(cur_agent.edge_list) + 1

        # print("Traversal parent of ", cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id], " is None")
        while len(traversal_queue) > 0 and not (self.sym_level == 2 and cur_agent.reached_fixed_point):
            cur_initset, mode_id, depth = traversal_queue.popleft()
            if self.sym_level != 2:
                waypoint: Waypoint = cur_agent.mode_list[mode_id]
            else:
                waypoint: Waypoint = cur_agent.abs_mode_list[mode_id]
                if cur_initset is None:
                    cur_initset = cur_agent.uncovered_sets_per_mode[mode_id]

                print("Abstract mode ", mode_id, "is being visited with parent ", traversal_parent[(mode_id, depth)])
            visited[waypoint.id] = 1
            # print("length of traversal queue: ", len(traversal_queue))
            # cur_initset = per_mode_initset[mode_id]
            cur_tubesegment, transform_time, fixed_point = self.tube_master.get_tubesegment(cur_agent, cur_initset,
                                                                                            waypoint,
                                                                                            self.time_step)
            cur_agent.number_of_segments_computed += 1
            # visited[mode_id] = -1
            cur_agent.transform_time += transform_time
            print("Mode at depth ", depth, " tube is computed with waypoint: ", waypoint.mode_parameters, " and mode id: ", waypoint.id)
            if self.sym_level != 2:
                if len(self.get_segment_unsafeset_inter_list(cur_tubesegment, self.global_unsafe_set)) > 0:
                    print("Is segment safe? No.")
                else:
                    print("Is segment safe? Yes.")
            else:
                unsafe_inter_list = self.get_segment_unsafeset_inter_list(cur_tubesegment, waypoint.unsafeset_list)
                if len(unsafe_inter_list) > 0:
                    print("Is segment safe? No.")
                    cur_agent.is_safe = False
                    if refine_counter < self.refine_threshold:
                        refine_depth = depth
                        curr_mode_id = mode_id
                        mode_id_tobecalled = []
                        while not curr_mode_id is None:  ## while you didn't reach a node that you can refine, keep going up
                            # clean the caches of all its children, including itself, before refining it.
                            children_ids = [curr_mode_id]
                            cur_agent.clean_mode_caches(curr_mode_id)
                            # for abs_edge_id in cur_agent.abs_mode_neighbors[curr_mode_id]:
                            #    children_ids.append(cur_agent.abs_edge_list[abs_edge_id].dest)
                            #    cur_agent.clean_mode_caches(children_ids[-1])
                            for tubesegment in full_tube:
                                seg_mode_id = tubesegment.virtual_mode
                                if seg_mode_id in children_ids:
                                    cur_agent.abs_mode_reachset[seg_mode_id].extend(list(tubesegment.tube_list_rect))
                                    for tube in tubesegment.tube_list:
                                        cur_agent.abs_mode_initset[seg_mode_id] = PolyUtils.get_region_union(
                                            cur_agent.abs_mode_initset[seg_mode_id], tube[0])
                            for tubesegment in full_tube:
                                for ind, abs_edge_id in enumerate(
                                        cur_agent.abs_mode_neighbors[tubesegment.virtual_mode]):
                                    next_mode_id = cur_agent.abs_edge_list[abs_edge_id].dest
                                    for next_initset in tubesegment.next_initset:
                                        if next_initset[1] == next_mode_id:
                                            for child_id in children_ids:
                                                if child_id == next_mode_id:
                                                    cur_agent.uncovered_sets_per_mode[child_id] = \
                                                        PolyUtils.subtract_regions(next_initset[0],
                                                                                   cur_agent.abs_mode_initset[child_id])
                                                    break
                                            break

                            refined = cur_agent.refine(curr_mode_id, unsafe_inter_list)
                            if refined:
                                refine_counter += 1
                                new_mode_id = len(cur_agent.abs_mode_list) - 1
                                print("Adding a new mode: ", new_mode_id, " with original modes: ",
                                      cur_agent.abs_mode_to_mode[new_mode_id])
                                print("Repeating reachability computation of the parent and updating the cache.")
                                # traversal_parent = {cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id]: None}
                                # traversal_queue.clear()
                                cur_agent.is_safe = True
                                ### removing the last reachset segment from the caches
                                cur_agent.abs_mode_reachset[new_mode_id] = \
                                    copy.deepcopy(cur_agent.abs_mode_reachset[curr_mode_id])
                                cur_agent.uncovered_sets_per_mode[new_mode_id] = \
                                    copy.deepcopy(cur_agent.uncovered_sets_per_mode[curr_mode_id])
                                cur_agent.abs_mode_initset[new_mode_id] = \
                                    copy.deepcopy(cur_agent.abs_mode_initset[curr_mode_id])
                                for seg_ind, tubesegment in enumerate(full_tube):
                                    still_neighbor = False
                                    for ind, abs_edge_id in enumerate(
                                            cur_agent.abs_mode_neighbors[tubesegment.virtual_mode]):
                                        dest_mode_id = cur_agent.abs_edge_list[abs_edge_id].dest
                                        if dest_mode_id == curr_mode_id:
                                            still_neighbor = True
                                    old_mode_idx = -1
                                    for ind_i, next_initset in enumerate(tubesegment.next_initset):
                                        if next_initset[1] == curr_mode_id:
                                            old_mode_idx = ind_i
                                            break
                                    if still_neighbor:
                                        full_tube[seg_ind].next_initset.append(
                                            [tubesegment.next_initset[old_mode_idx][0], new_mode_id])
                                    else:
                                        full_tube[seg_ind].next_initset[old_mode_idx][1] = new_mode_id



                                traversal_parent_id = traversal_parent[(curr_mode_id, refine_depth)]
                                if traversal_parent_id is None:
                                    # traversal_parent.pop((parent_mode_id, refine_depth))
                                    traversal_parent[(cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id], 0)] = None
                                    # traversal_queue.clear()
                                    traversal_queue.append(tuple([root_initset,
                                                                  cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id],
                                                                  0]))
                                else:
                                    for abs_edge_id in cur_agent.abs_mode_parents[curr_mode_id]:
                                        if cur_agent.abs_edge_list[abs_edge_id].source == traversal_parent_id:
                                            curr_min_index, curr_max_index, possible_next_initset_reg = \
                                                TubeMaster.tube_list_intersect_waypoint(cur_agent,
                                                                                        cur_agent.abs_mode_reachset[traversal_parent_id],
                                                                                        cur_agent.abs_edge_list[abs_edge_id])
                                            traversal_queue.append(tuple([possible_next_initset_reg, curr_mode_id,
                                                                          refine_depth]))
                                            traversal_parent[(curr_mode_id, refine_depth)] = traversal_parent_id
                                            print("Adding after refinment: traversal parent of ", curr_mode_id, " is ",
                                                  traversal_parent_id)
                                            break

                                    for abs_edge_id in cur_agent.abs_mode_parents[new_mode_id]:
                                        if cur_agent.abs_edge_list[abs_edge_id].source == traversal_parent_id:
                                            curr_min_index, curr_max_index, possible_next_initset_reg = \
                                                TubeMaster.tube_list_intersect_waypoint(cur_agent,
                                                                                        cur_agent.abs_mode_reachset[traversal_parent_id],
                                                                                        cur_agent.abs_edge_list[abs_edge_id])
                                            traversal_queue.append(tuple([possible_next_initset_reg, new_mode_id,
                                                                          refine_depth]))
                                            traversal_parent[(new_mode_id, refine_depth)] = traversal_parent_id
                                            print("Adding after refinment: traversal parent of ", new_mode_id, " is ",
                                                  traversal_parent_id)
                                            break
                                rev_ind = len(mode_id_tobecalled) - 1
                                while rev_ind > 0:
                                    refine_depth += 1
                                    traversal_queue.append(tuple([None, mode_id_tobecalled[rev_ind],
                                                                  refine_depth]))
                                    traversal_parent[(mode_id_tobecalled[rev_ind], refine_depth)] = curr_mode_id
                                    curr_mode_id = mode_id_tobecalled[rev_ind]
                                    rev_ind -= 1

                                #traversal_queue.append(tuple([root_initset,
                                #                              cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id],
                                #                              0]))
                                print("Done repeated reachability computation.")
                                break
                            # print("parent_mode_id", parent_mode_id)
                            mode_id_tobecalled.append(curr_mode_id)
                            curr_mode_id = traversal_parent[(curr_mode_id, refine_depth)]
                            unsafe_inter_list = []
                            full_tube_ind = len(full_tube) - 1
                            children_visited = {}
                            stop_loop = False
                            while full_tube_ind >= 0 and not stop_loop:
                                if full_tube[full_tube_ind].virtual_mode == curr_mode_id:
                                    stop_loop = True
                                if full_tube[full_tube_ind].virtual_mode in children_ids and\
                                        not full_tube[full_tube_ind].virtual_mode in children_visited:
                                    print("Deleting the latest tube segment in full tube of mode ",
                                          full_tube[full_tube_ind].virtual_mode, " at index ", full_tube_ind)
                                    full_tube.pop(full_tube_ind)
                                full_tube_ind -= 1
                            refine_depth -= 1


                        if refined:
                            continue
                        if curr_mode_id is None:
                            traversal_parent[(cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id], 0)] = None
                            # traversal_queue.clear()
                            traversal_queue.append(tuple([root_initset,
                                                          cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id],
                                                          0]))
                            rev_ind = len(mode_id_tobecalled) - 1
                            while rev_ind > 0:
                                refine_depth += 1
                                traversal_queue.append(tuple([None, mode_id_tobecalled[rev_ind],
                                                              refine_depth]))
                                traversal_parent[(mode_id_tobecalled[rev_ind], refine_depth)] = curr_mode_id
                                curr_mode_id = mode_id_tobecalled[rev_ind]
                                rev_ind -= 1
                        """
                        if depth > 0:
                            traversal_queue.extend([tuple([cur_agent.abs_mode_initset[mode_parents[mode_1][0].source],
                                                        mode_parents[mode_1][0], depth])])
                        else:
                            traversal_queue.extend([tuple([root_initset, mode_1, depth])])
                            traversal_queue.extend([tuple([root_initset, mode_2, depth])])
                        continue
                        """
                else:
                    print("Is segment safe? Yes.")

            print("Done checking safety")
            if self.sym_level == 2:
                cur_agent.reached_fixed_point = cur_agent.check_fixed_point()
                print("Done checking fixed point")
            full_tube.append(cur_tubesegment)
            if (len(cur_tubesegment.guard_max_index) == 0 or max(cur_tubesegment.guard_max_index) == -1) \
                    and ((self.sym_level != 2 and len(cur_agent.mode_neighbors[waypoint.id]) > 0)
                         or (self.sym_level == 2 and len(cur_agent.abs_mode_neighbors[waypoint.id]) > 0)):

                print(PolyUtils.get_bounding_box(cur_tubesegment.tube_list[-1][-1]))
                print("tube segment does not reach guard")
                break
            # look_back.append([look_back[-1][0] + len(cur_tubesegment.tube_list[-1]), max(cur_tubesegment.guard_max_index)
            #                  - min(cur_tubesegment.guard_min_index)])
            if self.sym_level != 2:
                neighbors = cur_agent.mode_neighbors[mode_id]
            else:
                neighbors = cur_agent.abs_mode_neighbors[mode_id]
            for ind, edge_id in enumerate(neighbors):
                time_passed: float = time_passed + min(cur_tubesegment.guard_min_index) * self.time_step
                if self.sym_level != 2:
                    next_mode_id = cur_agent.edge_list[edge_id].dest
                else:
                    next_mode_id = cur_agent.abs_edge_list[edge_id].dest
                # if visited[next_mode_id] == -1:
                # per_mode_initset[next_mode_id] = cur_tubesegment.next_initset[ind]
                # visited[next_mode_id] = 1
                if self.sym_level == 2:
                    for next_initset in cur_tubesegment.next_initset:
                        if next_initset[1] == next_mode_id and not pc.is_empty(next_initset[0]):
                            traversal_queue.extend([tuple([next_initset[0], next_mode_id, depth + 1])])
                            print("Traversal parent of ", next_mode_id, " is ", mode_id)
                            traversal_parent[(next_mode_id, depth + 1)] = mode_id
                else:
                    if not pc.is_empty(cur_tubesegment.next_initset[ind]):
                        traversal_queue.extend([tuple([cur_tubesegment.next_initset[ind], next_mode_id, depth + 1])])
                        print("Traversal parent of ", next_mode_id, " is ", mode_id)

                #else:
                #    pdb.set_trace()
                # else:
                #    per_mode_initset[next_mode_id] = pc.union(
                #        cur_tubesegment.next_initset[ind], per_mode_initset[next_mode_id])
        #################################
        print("execution time before transform back to original tube : ", (time.time() - t) / 60.0)
        if self.sym_level == 2:
            print("Total number of refinements: ", refine_counter)
            full_tube = []
            traversal_queue = deque([cur_agent.initial_mode_id])
            while len(traversal_queue) > 0:
                mode_id = traversal_queue.popleft()
                abs_reachset = cur_agent.abs_mode_reachset[cur_agent.mode_to_abs_mode[mode_id]]
                if len(abs_reachset) != 0:
                    transform_information = cur_agent.get_transform_information(
                        cur_agent.mode_list[mode_id])
                    tube_list = []
                    rect_tube_list = []
                    for tube in abs_reachset:
                        tube_list.append([cur_agent.transform_poly_from_virtual(pc.box2poly(tube[i][:, :].T),
                                                                                transform_information)
                                          for i in range(len(tube))])
                        rect_tube_list.append([])
                        for poly in tube_list[-1]:
                            rect_tube_list.append(PolyUtils.get_bounding_box(poly))
                    reachset_segment = ReachtubeSegment(tube_list, rect_tube_list, None, -1,
                                                        -1, np.array([]), cur_agent.mode_to_abs_mode[mode_id])
                    cur_agent.number_of_segments_transformed += 1
                    cur_agent.number_of_tubes_transformed += len(reachset_segment.tube_list)
                    full_tube.append(reachset_segment)
                    for ind, edge_id in enumerate(cur_agent.mode_neighbors[mode_id]):
                        traversal_queue.extend([cur_agent.edge_list[edge_id].dest])
                else:
                    print("abs_reachset empty", mode_id, cur_agent.mode_to_abs_mode[mode_id])
                    # pdb.set_trace()
        '''
        for i in range(len(cur_agent.mode_list)):
            waypoint: Waypoint = cur_agent.mode_list[i]
            cur_tubesegment, transform_time, fixed_point = self.tube_master.get_tubesegment(cur_agent, cur_initset,
                                                                                            waypoint,
                                                                                            self.time_step)

            cur_agent.transform_time += transform_time
            print("Segment ", i, " tube is computed with waypoint: ", waypoint.mode_parameters)
            if (len(cur_tubesegment.guard_max_index) == 0 or max(cur_tubesegment.guard_max_index) == -1)\
                and len(cur_agent.mode_neighbors[waypoint.id]) > 0:
                print("tube segment does not reach guard")
                break
            if len(cur_agent.mode_neighbors[waypoint.id]) > 0:
                look_back.append([look_back[-1][0] + len(cur_tubesegment.tube), max(cur_tubesegment.guard_max_index)
                                  - min(cur_tubesegment.guard_min_index)])
                cur_initset: pc.Polytope = cur_tubesegment.next_initset[0]
                time_passed: float = time_passed + min(cur_tubesegment.guard_min_index) * self.time_step

            full_tube.append(cur_tubesegment)
            # TAC, uncomment after
            #if not fixed_point:
            #    cur_initset: pc.Polytope = pc.intersect(cur_tubesegment.next_initset,
            #                                            pc.box2poly(waypoint.original_guard.T))
        '''
        return full_tube, [], look_back

    def compute_agent_full_tube_bfs(self, cur_agent: Agent, t = 0) -> Tuple[List[ReachtubeSegment], List[List[int]]]:
        full_tube: List[ReachtubeSegment] = []
        look_back: List[List[int]] = [[0, 0]]
        # list of pairs of index of a jump, look back number of indices
        # all polytopes in the tube between two consecutive indices would have a lookback equal to the previous one
        time_passed = 0
        visited = {}
        traversal_parent = {}
        # TODO update this loop to use the transform class
        #################################
        transform_information = cur_agent.get_transform_information(cur_agent.mode_list[cur_agent.initial_mode_id])
        root_initset: pc.Region = cur_agent.transform_poly_to_virtual(cur_agent.initial_set, transform_information)
        cur_initset: pc.Region = root_initset
        traversal_queue = deque([tuple([cur_initset, cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id], 0])])
        refine_counter = 0
        traversal_parent[(cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id], 0)] = None

        while len(traversal_queue) > 0 and not (self.sym_level == 2 and cur_agent.reached_fixed_point):
        # while len(traversal_queue) > 0:
        # Get current initial set Kv
            cur_initset, mode_id, depth = traversal_queue.popleft()
            waypoint: Waypoint = cur_agent.abs_mode_list[mode_id]
            if cur_initset is None:
                cur_initset = cur_agent.uncovered_sets_per_mode[mode_id]

            print("Abstract mode ", mode_id, "is being visited with parent ", traversal_parent[(mode_id, depth)])
            visited[waypoint.id] = 1
            # The get tubesegment can provide intersection between reachtube and guard
            cur_tubesegment, transform_time, fixed_point = self.tube_master.get_tubesegment(cur_agent, cur_initset,
                                                                                            waypoint,
                                                                                            self.time_step)
            cur_agent.number_of_segments_computed += 1
            # visited[mode_id] = -1
            cur_agent.transform_time += transform_time
            print("Mode at depth ", depth, " tube is computed with waypoint: ", waypoint.mode_parameters, " and mode id: ", waypoint.id)

            unsafe_inter_list = self.get_segment_unsafeset_inter_list(cur_tubesegment, waypoint.unsafeset_list)
            if len(unsafe_inter_list) > 0:
                print("Is segment safe? No.")
                cur_agent.is_safe = False
                if refine_counter < self.refine_threshold:
                    refine_depth = depth
                    curr_mode_id = mode_id
                    mode_id_tobecalled = []
                    while not curr_mode_id is None:  ## while you didn't reach a node that you can refine, keep going up
                        # clean the caches of all its children, including itself, before refining it.
                        children_ids = [curr_mode_id]
                        cur_agent.clean_mode_caches(curr_mode_id)
                        # for abs_edge_id in cur_agent.abs_mode_neighbors[curr_mode_id]:
                        #    children_ids.append(cur_agent.abs_edge_list[abs_edge_id].dest)
                        #    cur_agent.clean_mode_caches(children_ids[-1])

                        # Get additional initial sets
                        for tubesegment in full_tube:
                            seg_mode_id = tubesegment.virtual_mode
                            if seg_mode_id in children_ids:
                                cur_agent.abs_mode_reachset[seg_mode_id].extend(list(tubesegment.tube_list_rect))
                                for tube in tubesegment.tube_list:
                                    cur_agent.abs_mode_initset[seg_mode_id] = PolyUtils.get_region_union(
                                        cur_agent.abs_mode_initset[seg_mode_id], tube[0])
                        for tubesegment in full_tube:
                            for ind, abs_edge_id in enumerate(
                                    cur_agent.abs_mode_neighbors[tubesegment.virtual_mode]):
                                next_mode_id = cur_agent.abs_edge_list[abs_edge_id].dest
                                for next_initset in tubesegment.next_initset:
                                    if next_initset[1] == next_mode_id:
                                        for child_id in children_ids:
                                            if child_id == next_mode_id:
                                                cur_agent.uncovered_sets_per_mode[child_id] = \
                                                    PolyUtils.subtract_regions(next_initset[0],
                                                                                cur_agent.abs_mode_initset[child_id])
                                                break
                                        break

                        # Probably the split mode function?
                        refined = cur_agent.refine(curr_mode_id, unsafe_inter_list)
                        if refined:
                            refine_counter += 1
                            new_mode_id = len(cur_agent.abs_mode_list) - 1
                            print("Adding a new mode: ", new_mode_id, " with original modes: ",
                                    cur_agent.abs_mode_to_mode[new_mode_id])
                            print("Repeating reachability computation of the parent and updating the cache.")
                            # traversal_parent = {cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id]: None}
                            # traversal_queue.clear()
                            cur_agent.is_safe = True
                            ### removing the last reachset segment from the caches
                            cur_agent.abs_mode_reachset[new_mode_id] = \
                                copy.deepcopy(cur_agent.abs_mode_reachset[curr_mode_id])
                            cur_agent.uncovered_sets_per_mode[new_mode_id] = \
                                copy.deepcopy(cur_agent.uncovered_sets_per_mode[curr_mode_id])
                            cur_agent.abs_mode_initset[new_mode_id] = \
                                copy.deepcopy(cur_agent.abs_mode_initset[curr_mode_id])
                            for seg_ind, tubesegment in enumerate(full_tube):
                                still_neighbor = False
                                for ind, abs_edge_id in enumerate(
                                        cur_agent.abs_mode_neighbors[tubesegment.virtual_mode]):
                                    dest_mode_id = cur_agent.abs_edge_list[abs_edge_id].dest
                                    if dest_mode_id == curr_mode_id:
                                        still_neighbor = True
                                old_mode_idx = -1
                                for ind_i, next_initset in enumerate(tubesegment.next_initset):
                                    if next_initset[1] == curr_mode_id:
                                        old_mode_idx = ind_i
                                        break
                                if still_neighbor:
                                    full_tube[seg_ind].next_initset.append(
                                        [tubesegment.next_initset[old_mode_idx][0], new_mode_id])
                                else:
                                    full_tube[seg_ind].next_initset[old_mode_idx][1] = new_mode_id



                            traversal_parent_id = traversal_parent[(curr_mode_id, refine_depth)]
                            if traversal_parent_id is None:
                                # traversal_parent.pop((parent_mode_id, refine_depth))
                                traversal_parent[(cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id], 0)] = None
                                # traversal_queue.clear()
                                traversal_queue.append(tuple([root_initset,
                                                                cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id],
                                                                0]))
                            else:
                                for abs_edge_id in cur_agent.abs_mode_parents[curr_mode_id]:
                                    if cur_agent.abs_edge_list[abs_edge_id].source == traversal_parent_id:
                                        curr_min_index, curr_max_index, possible_next_initset_reg = \
                                            TubeMaster.tube_list_intersect_waypoint(cur_agent,
                                                                                    cur_agent.abs_mode_reachset[traversal_parent_id],
                                                                                    cur_agent.abs_edge_list[abs_edge_id])
                                        traversal_queue.append(tuple([possible_next_initset_reg, curr_mode_id,
                                                                        refine_depth]))
                                        traversal_parent[(curr_mode_id, refine_depth)] = traversal_parent_id
                                        print("Adding after refinment: traversal parent of ", curr_mode_id, " is ",
                                                traversal_parent_id)
                                        break

                                for abs_edge_id in cur_agent.abs_mode_parents[new_mode_id]:
                                    if cur_agent.abs_edge_list[abs_edge_id].source == traversal_parent_id:
                                        curr_min_index, curr_max_index, possible_next_initset_reg = \
                                            TubeMaster.tube_list_intersect_waypoint(cur_agent,
                                                                                    cur_agent.abs_mode_reachset[traversal_parent_id],
                                                                                    cur_agent.abs_edge_list[abs_edge_id])
                                        traversal_queue.append(tuple([possible_next_initset_reg, new_mode_id,
                                                                        refine_depth]))
                                        traversal_parent[(new_mode_id, refine_depth)] = traversal_parent_id
                                        print("Adding after refinment: traversal parent of ", new_mode_id, " is ",
                                                traversal_parent_id)
                                        break
                            rev_ind = len(mode_id_tobecalled) - 1
                            while rev_ind > 0:
                                refine_depth += 1
                                traversal_queue.append(tuple([None, mode_id_tobecalled[rev_ind],
                                                                refine_depth]))
                                traversal_parent[(mode_id_tobecalled[rev_ind], refine_depth)] = curr_mode_id
                                curr_mode_id = mode_id_tobecalled[rev_ind]
                                rev_ind -= 1

                            #traversal_queue.append(tuple([root_initset,
                            #                              cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id],
                            #                              0]))
                            print("Done repeated reachability computation.")
                            break
                        # print("parent_mode_id", parent_mode_id)
                        mode_id_tobecalled.append(curr_mode_id)
                        curr_mode_id = traversal_parent[(curr_mode_id, refine_depth)]
                        unsafe_inter_list = []
                        full_tube_ind = len(full_tube) - 1
                        children_visited = {}
                        stop_loop = False
                        while full_tube_ind >= 0 and not stop_loop:
                            if full_tube[full_tube_ind].virtual_mode == curr_mode_id:
                                stop_loop = True
                            if full_tube[full_tube_ind].virtual_mode in children_ids and\
                                    not full_tube[full_tube_ind].virtual_mode in children_visited:
                                print("Deleting the latest tube segment in full tube of mode ",
                                        full_tube[full_tube_ind].virtual_mode, " at index ", full_tube_ind)
                                full_tube.pop(full_tube_ind)
                            full_tube_ind -= 1
                        refine_depth -= 1


                    if refined:
                        continue
                    if curr_mode_id is None:
                        traversal_parent[(cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id], 0)] = None
                        # traversal_queue.clear()
                        traversal_queue.append(tuple([root_initset,
                                                        cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id],
                                                        0]))
                        rev_ind = len(mode_id_tobecalled) - 1
                        while rev_ind > 0:
                            refine_depth += 1
                            traversal_queue.append(tuple([None, mode_id_tobecalled[rev_ind],
                                                            refine_depth]))
                            traversal_parent[(mode_id_tobecalled[rev_ind], refine_depth)] = curr_mode_id
                            curr_mode_id = mode_id_tobecalled[rev_ind]
                            rev_ind -= 1
                    """
                    if depth > 0:
                        traversal_queue.extend([tuple([cur_agent.abs_mode_initset[mode_parents[mode_1][0].source],
                                                    mode_parents[mode_1][0], depth])])
                    else:
                        traversal_queue.extend([tuple([root_initset, mode_1, depth])])
                        traversal_queue.extend([tuple([root_initset, mode_2, depth])])
                    continue
                    """
            else:
                print("Is segment safe? Yes.")

            print("Done checking safety")

            cur_agent.reached_fixed_point = cur_agent.check_fixed_point()
            print("Done checking fixed point")
            full_tube.append(cur_tubesegment)
            if (len(cur_tubesegment.guard_max_index) == 0 or max(cur_tubesegment.guard_max_index) == -1) \
                    and ((self.sym_level != 2 and len(cur_agent.mode_neighbors[waypoint.id]) > 0)
                         or (self.sym_level == 2 and len(cur_agent.abs_mode_neighbors[waypoint.id]) > 0)):

                print(PolyUtils.get_bounding_box(cur_tubesegment.tube_list[-1][-1]))
                print("tube segment does not reach guard")
                break
            # look_back.append([look_back[-1][0] + len(cur_tubesegment.tube_list[-1]), max(cur_tubesegment.guard_max_index)
            #                  - min(cur_tubesegment.guard_min_index)])
            neighbors = cur_agent.abs_mode_neighbors[mode_id]
            for ind, edge_id in enumerate(neighbors):
                time_passed: float = time_passed + min(cur_tubesegment.guard_min_index) * self.time_step
                next_mode_id = cur_agent.abs_edge_list[edge_id].dest
                # if visited[next_mode_id] == -1:
                # per_mode_initset[next_mode_id] = cur_tubesegment.next_initset[ind]
                # visited[next_mode_id] = 1
                for next_initset in cur_tubesegment.next_initset:
                    if next_initset[1] == next_mode_id and not pc.is_empty(next_initset[0]):
                        traversal_queue.extend([tuple([next_initset[0], next_mode_id, depth + 1])])
                        print("Traversal parent of ", next_mode_id, " is ", mode_id)
                        traversal_parent[(next_mode_id, depth + 1)] = mode_id
                #else:
                #    pdb.set_trace()
                # else:
                #    per_mode_initset[next_mode_id] = pc.union(
                #        cur_tubesegment.next_initset[ind], per_mode_initset[next_mode_id])
        #################################
        print("execution time before transform back to original tube : ", (time.time() - t) / 60.0)
        print("Total number of refinements: ", refine_counter)
        full_tube = []
        traversal_queue = deque([cur_agent.initial_mode_id])
        while len(traversal_queue) > 0:
            mode_id = traversal_queue.popleft()
            abs_reachset = cur_agent.abs_mode_reachset[cur_agent.mode_to_abs_mode[mode_id]]
            if len(abs_reachset) != 0:
                transform_information = cur_agent.get_transform_information(
                    cur_agent.mode_list[mode_id])
                tube_list = []
                rect_tube_list = []
                for tube in abs_reachset:
                    tube_list.append([cur_agent.transform_poly_from_virtual(pc.box2poly(tube[i][:, :].T),
                                                                            transform_information)
                                        for i in range(len(tube))])
                    rect_tube_list.append([])
                    for poly in tube_list[-1]:
                        rect_tube_list.append(PolyUtils.get_bounding_box(poly))
                reachset_segment = ReachtubeSegment(tube_list, rect_tube_list, None, -1,
                                                    -1, np.array([]), cur_agent.mode_to_abs_mode[mode_id])
                cur_agent.number_of_segments_transformed += 1
                cur_agent.number_of_tubes_transformed += len(reachset_segment.tube_list)
                full_tube.append(reachset_segment)
                for ind, edge_id in enumerate(cur_agent.mode_neighbors[mode_id]):
                    traversal_queue.extend([cur_agent.edge_list[edge_id].dest])
            else:
                print("abs_reachset empty", mode_id, cur_agent.mode_to_abs_mode[mode_id])
                # pdb.set_trace()
        '''
        for i in range(len(cur_agent.mode_list)):
            waypoint: Waypoint = cur_agent.mode_list[i]
            cur_tubesegment, transform_time, fixed_point = self.tube_master.get_tubesegment(cur_agent, cur_initset,
                                                                                            waypoint,
                                                                                            self.time_step)

            cur_agent.transform_time += transform_time
            print("Segment ", i, " tube is computed with waypoint: ", waypoint.mode_parameters)
            if (len(cur_tubesegment.guard_max_index) == 0 or max(cur_tubesegment.guard_max_index) == -1)\
                and len(cur_agent.mode_neighbors[waypoint.id]) > 0:
                print("tube segment does not reach guard")
                break
            if len(cur_agent.mode_neighbors[waypoint.id]) > 0:
                look_back.append([look_back[-1][0] + len(cur_tubesegment.tube), max(cur_tubesegment.guard_max_index)
                                  - min(cur_tubesegment.guard_min_index)])
                cur_initset: pc.Polytope = cur_tubesegment.next_initset[0]
                time_passed: float = time_passed + min(cur_tubesegment.guard_min_index) * self.time_step

            full_tube.append(cur_tubesegment)
            # TAC, uncomment after
            #if not fixed_point:
            #    cur_initset: pc.Polytope = pc.intersect(cur_tubesegment.next_initset,
            #                                            pc.box2poly(waypoint.original_guard.T))
        '''
        return full_tube, [], look_back

    def SymAR(self, cur_agent: Agent, pv_s, parent, Kv, Rv, refine_depth, traversal_parent, full_tube, depth):
        print("depth", depth,"refine_depth", refine_depth)
        print(f"parent of {pv_s} is {parent}")
        # print("Initset",Kv)

        for key in cur_agent.abs_mode_reachset:
            print(f"abs_mode {key} len {len(cur_agent.abs_mode_reachset[key])}")
            if len(cur_agent.abs_mode_reachset[key])>15:
                raise TimeoutError

        if refine_depth > cur_agent.max_refinements:
            cur_agent.max_refinements = refine_depth
        # Kv = pc.Region()
        # if parent is None:
        #     transform_information = cur_agent.get_transform_information(cur_agent.mode_list[cur_agent.initial_mode_id])
        #     root_initset: pc.Region = cur_agent.transform_poly_to_virtual(cur_agent.initial_set, transform_information)
        #     Kv: pc.Region = root_initset
        # else:
        #     # for abs_edge_id in cur_agent.abs_mode_parents[pv_s]:
        #     #     if cur_agent.abs_edge_list[abs_edge_id].source == parent:
        #
        #     # Kv = Rv.next_initset
        #     for abs_edge_id in cur_agent.abs_mode_parents[pv_s]:
        #         if cur_agent.abs_edge_list[abs_edge_id].source == parent:
        #             curr_min_index, curr_max_index, possible_next_initset_reg = \
        #                 TubeMaster.tube_list_intersect_waypoint(cur_agent,
        #                                                         cur_agent.abs_mode_reachset[parent],
        #                                                         cur_agent.abs_edge_list[abs_edge_id])
        #             Kv = possible_next_initset_reg

            # neighbors = cur_agent.abs_mode_neighbors[parent]
            # for ind, edge_id in enumerate(neighbors):
            #     # time_passed: float = time_passed + min(Rv.guard_min_index) * self.time_step
            #     for next_initset in Rv.next_initset:
            #         if next_initset[1] == pv_s and not pc.is_empty(next_initset[0]):
            #             Kv = next_initset[0]
            #             break
            # Kv = cur_agent.uncovered_sets_per_mode[pv_s]
        waypoint: Waypoint = cur_agent.abs_mode_list[pv_s]

        # Check if Kv already in cache
        # if ((not pc.is_empty(cur_agent.abs_mode_initset[waypoint.id]))\
        #             and pc.is_subset(Kv, cur_agent.abs_mode_initset[waypoint.id])):
        if cur_agent.reached_fixed_point:
            min_index = []
            max_index = []
            next_initset = []
            tube_list = cur_agent.abs_mode_reachset[waypoint.id]
            trace_list = None
            num_neighbors = len(cur_agent.abs_mode_neighbors[waypoint.id])
            result_tube_list = []
            for tube in tube_list:
                result_tube_list.append([pc.box2poly(tube[i][:, :].T) for i in range(len(tube))])
            transform_time = 0
            tube_segment = ReachtubeSegment(result_tube_list, tube_list, None, min_index, max_index, next_initset, waypoint.id)
            full_tube.append(tube_segment)
            print("reaching one fixed point")
            return cur_agent, {}, True, True, full_tube

        if pc.is_empty(Kv):
            print("Guard not reached, return")
            return cur_agent, {}, True, True, full_tube

        Av_p = copy.deepcopy(cur_agent)
        refine = False
        reftree = {}
        for mode in cur_agent.abs_mode_list:
            reftree[mode.id] = [mode.id]
        Rv_p, transform_time, fixed_point = self.tube_master.get_tubesegment(Av_p, Kv,
                                                                                        waypoint,
                                                                                        self.time_step)
        Av_p.number_of_segments_computed += 1
        # visited[mode_id] = -1
        Av_p.transform_time += transform_time

        print("Mode at depth ", depth, " tube is computed with waypoint: ", waypoint.mode_parameters, " and mode id: ", waypoint.id)

        if depth > Av_p.max_depth:
            print("Reaching max branch depth, return")

        unsafe_inter_list = self.get_segment_unsafeset_inter_list(Rv_p, waypoint.unsafeset_list)
        if len(unsafe_inter_list) > 0:
            refine = True
        else:
            print("curr mode safe, checking next mode")
            fixed_point = True
            # print(Rv_p.next_initset)
            to_reach_queue = copy.deepcopy(Av_p.abs_mode_neighbors[pv_s])
            reached_list = []
            while to_reach_queue:
                edge_ind = to_reach_queue.pop(0)
                reached_list.append(edge_ind)
                pv_p = Av_p.abs_edge_list[edge_ind].dest
                # for pv_p in reftree[next_mode_id]:
                print(f"potential next mode id {pv_p}, current mode id {pv_s}")

            # for ind, edge_id in enumerate(neighbors):
            #     time_passed: float = time_passed + min(Rv_p.guard_min_index) * self.time_step
            #     next_mode_id = cur_agent.abs_edge_list[edge_id].dest
                # if visited[next_mode_id] == -1:
                # per_mode_initset[next_mode_id] = cur_tubesegment.next_initset[ind]
                # visited[next_mode_id] = 1
                Kv = pc.Region()
                # for next_initset in Rv_p.next_initset:
                #     if next_initset[1] == pv_p and not pc.is_empty(next_initset[0]):
                #         # traversal_queue.extend([tuple([next_initset[0], next_mode_id, depth + 1])])
                #         # print("Traversal parent of ", next_mode_id, " is ", mode_id)
                #         # traversal_parent[(next_mode_id, depth + 1)] = mode_id
                #         Kv = next_initset[0]

                curr_min_index, curr_max_index, possible_next_initset_reg = \
                    TubeMaster.tube_list_intersect_waypoint(Av_p,
                                                            Rv_p.tube_list_rect,
                                                            Av_p.abs_edge_list[edge_ind])
                if type(possible_next_initset_reg) == pc.Polytope:
                    possible_next_initset_reg = pc.Region(list_poly=[possible_next_initset_reg])

                for new_poly in possible_next_initset_reg.list_poly:
                    rect_contain = False
                    if type(Kv) == pc.Polytope:
                        Kv = pc.Region(list_poly=[Kv])
                    for old_poly in Kv.list_poly:
                        if PolyUtils.does_rect_contain(pc.bounding_box(new_poly), pc.bounding_box(old_poly)):
                            rect_contain = True
                            break
                    if not rect_contain:
                        Kv = PolyUtils.get_region_union(Kv, new_poly)

                Av_p.check_fixed_point()
                cached_initset = Av_p.abs_mode_initset[pv_p]
                if type(cached_initset) == pc.Polytope:
                    cached_initset = pc.Region(list_poly=[cached_initset])
                if type(Kv) == pc.Polytope:
                    Kv = pc.Region(list_poly=[Kv])

                if not cached_initset.list_poly:
                    contain = False
                else:
                    contain = False
                    for poly in cached_initset.list_poly:
                        for new_poly in Kv.list_poly:
                            poly_box = pc.bounding_box(poly)
                            new_poly_box = pc.bounding_box(new_poly)
                            if self.tmp:
                                if PolyUtils.does_rect_contain(new_poly_box, poly_box):
                                    contain = True
                            else:
                                if PolyUtils.does_rect_contain(poly_box, new_poly_box):
                                    contain = True
                            
                if contain:
                    print(f"One fixed point reached {pv_s}")
                    curfix = True
                    cursafe = True
                    curreftree = {}
                else:
                    Av_p, curreftree, curfix, cursafe, full_tube = \
                        self.SymAR(Av_p, pv_p, pv_s, Kv, Rv_p, refine_depth, traversal_parent, full_tube, depth+1)

                for tmp in Av_p.abs_mode_neighbors[pv_s]:
                    if tmp not in to_reach_queue and tmp not in reached_list:
                        to_reach_queue.append(tmp)

                if curreftree:
                    if isinstance(curreftree[0][0],Waypoint):
                        print("here")
                if not curfix:
                    fixed_point = False
                for key in curreftree:
                    if key in reftree:
                        for element in curreftree[key]:
                            if element not in reftree[key]:
                                reftree[key].append(element)
                    else:
                        reftree[key] = curreftree[key]
                if not cursafe:
                    if not self.safe_plot:
                        Av_p = copy.deepcopy(cur_agent)
                    reftree = {}
                    for mode in Av_p.abs_mode_list:
                        reftree[mode.id] = [mode.id]
                    refine = True
                    break
                if refine:
                    break
        if refine:
            if refine_depth < self.refine_threshold:
                Av_p.clean_mode_caches(pv_s)
                if len(Av_p.abs_mode_to_mode[pv_s]) <= 1:
                    print(f"Mode unsafe {pv_s}, return")
                    return Av_p, reftree, False, False, full_tube
                if pv_s == 6:
                    print("Stop here")
                # refined = Av_p.refine_noreplace(pv_s, unsafe_inter_list)
                Av_p.num_refinements += 1
                refined = Av_p.refine(pv_s, unsafe_inter_list)
                if not refined:
                    return Av_p, reftree, False, False, full_tube
                pv1 = pv_s
                pv2 = len(Av_p.abs_mode_list) - 1
                print("Adding a new mode: ", pv1, " with original modes: ",
                      Av_p.abs_mode_to_mode[pv1])
                print("Adding a new mode: ", pv2, " with original modes: ",
                        Av_p.abs_mode_to_mode[pv2])
                reftree[pv_s] = [pv1, pv2]

                # Line 36-37
                # Av_p.abs_mode_reachset[pv1] = \
                #     copy.deepcopy(cur_agent.abs_mode_reachset[pv_s])
                # Av_p.uncovered_sets_per_mode[pv1] = \
                #     copy.deepcopy(cur_agent.uncovered_sets_per_mode[pv_s])
                # Av_p.abs_mode_initset[pv1] = \
                #     copy.deepcopy(cur_agent.abs_mode_initset[pv_s])
                #
                # Av_p.abs_mode_reachset[pv2] = \
                #     copy.deepcopy(cur_agent.abs_mode_reachset[pv_s])
                # Av_p.uncovered_sets_per_mode[pv2] = \
                #     copy.deepcopy(cur_agent.uncovered_sets_per_mode[pv_s])
                # Av_p.abs_mode_initset[pv2] = \
                #     copy.deepcopy(cur_agent.abs_mode_initset[pv_s])

                # Av_p.abs_mode_reachset[pv1] = []
                # Av_p.uncovered_sets_per_mode[pv1] = []
                # Av_p.abs_mode_initset[pv1] = []
                #
                # Av_p.abs_mode_reachset[pv2] = []
                # Av_p.uncovered_sets_per_mode[pv2] = []
                # Av_p.abs_mode_initset[pv2] = []
                if parent is None or pv_s == Av_p.mode_to_abs_mode[Av_p.initial_mode_id]:
                    transform_information = Av_p.get_transform_information(\
                        Av_p.mode_list[Av_p.initial_mode_id])
                    root_initset: pc.Region = Av_p.transform_poly_to_virtual(Av_p.initial_set,\
                                                                                transform_information)
                    return self.SymAR(Av_p,Av_p.mode_to_abs_mode[Av_p.initial_mode_id],\
                        None, root_initset, None, refine_depth+1, traversal_parent, full_tube, depth)
                else:
                    fixed_point = True
                    # have_ancester = False
                    for edge_ind in Av_p.abs_mode_parents[pv_s]:
                        if Av_p.abs_edge_list[edge_ind].source != pv_s:
                            # have_ancester = True
                            for pv_p in reftree[pv_s]:
                                ancester_mode = Av_p.abs_edge_list[edge_ind].source
                                Kv = pc.Region()
                                # for next_initset in Rv_p.next_initset:
                                #     if next_initset[1] == pv_p and not pc.is_empty(next_initset[0]):
                                #         # traversal_queue.extend([tuple([next_initset[0], next_mode_id, depth + 1])])
                                #         # print("Traversal parent of ", next_mode_id, " is ", mode_id)
                                #         # traversal_parent[(next_mode_id, depth + 1)] = mode_id
                                #         Kv = next_initset[0]
                                for parent_edge_ind in Av_p.abs_mode_neighbors[ancester_mode]:
                                    if Av_p.abs_edge_list[parent_edge_ind].dest == pv_p:
                                        curr_min_index, curr_max_index, possible_next_initset_reg = \
                                            TubeMaster.tube_list_intersect_waypoint(Av_p,
                                                                                    Av_p.abs_mode_reachset[ancester_mode],
                                                                                    Av_p.abs_edge_list[edge_ind])
                                        if type(possible_next_initset_reg) == pc.Polytope:
                                            possible_next_initset_reg = pc.Region(list_poly=[possible_next_initset_reg])

                                        for new_poly in possible_next_initset_reg.list_poly:
                                            rect_contain = False
                                            if type(Kv) == pc.Polytope:
                                                Kv = pc.Region(list_poly=[Kv])
                                            for old_poly in Kv.list_poly:
                                                if PolyUtils.does_rect_contain(pc.bounding_box(new_poly), pc.bounding_box(old_poly)):
                                                    rect_contain = True
                                                    break
                                            if not rect_contain:
                                                Kv = PolyUtils.get_region_union(Kv, new_poly)
                                # Kv = possible_next_initset_reg

                                Av_p, curreftree, curfix, cursafe, full_tube = \
                                    self.SymAR(Av_p, pv_p, parent, Kv, Rv, refine_depth+1, traversal_parent, full_tube, depth)
                                if curfix == False:
                                    fixed_point = False
                                for key in curreftree:
                                    if key in reftree:
                                        for element in curreftree[key]:
                                            if element not in reftree[key]:
                                                reftree[key].append(element)
                                    else:
                                        reftree[key] = curreftree[key]
                                if not cursafe:
                                    return Av_p, reftree, False, False, full_tube
                    # if not have_ancester:
                    #     transform_information = Av_p.get_transform_information( \
                    #         Av_p.mode_list[Av_p.initial_mode_id])
                    #     root_initset: pc.Region = Av_p.transform_poly_to_virtual(Av_p.initial_set, \
                    #                                                              transform_information)
                    #     return self.SymAR(Av_p, Av_p.mode_to_abs_mode[Av_p.initial_mode_id], \
                    #                       None, root_initset, None, refine_depth + 1, traversal_parent, full_tube,
                    #                       depth)
            else:
                print("reaching refine threshold, return")
                if not self.safe_plot:
                    raise TimeoutError
                fixed_point = True
                print(Rv_p.next_initset)
                for edge_ind in Av_p.abs_mode_neighbors[pv_s]:
                    pv_p = Av_p.abs_edge_list[edge_ind].dest
                    print(f"potential next mode id {pv_p}, current mode id {pv_s}")
                    # for pv_p in reftree[next_mode_id]:
                    neighbors = Av_p.abs_mode_neighbors[pv_s]
                    # for ind, edge_id in enumerate(neighbors):
                    #     time_passed: float = time_passed + min(Rv_p.guard_min_index) * self.time_step
                    #     next_mode_id = cur_agent.abs_edge_list[edge_id].dest
                    # if visited[next_mode_id] == -1:
                    # per_mode_initset[next_mode_id] = cur_tubesegment.next_initset[ind]
                    # visited[next_mode_id] = 1
                    Kv = pc.Region()
                    for next_initset in Rv_p.next_initset:
                        if next_initset[1] == pv_p and not pc.is_empty(next_initset[0]):
                            # traversal_queue.extend([tuple([next_initset[0], next_mode_id, depth + 1])])
                            # print("Traversal parent of ", next_mode_id, " is ", mode_id)
                            # traversal_parent[(next_mode_id, depth + 1)] = mode_id
                            Kv = next_initset[0]

                    # traversal_parent[(pv_p, refine_depth)] = pv_s
                    Av_p.reached_fixed_point = Av_p.check_fixed_point()
                    cached_initset = Av_p.abs_mode_initset[pv_p]
                    if type(cached_initset) == pc.Polytope:
                        cached_initset = pc.Region(list_poly=[cached_initset])
                    if type(Kv) == pc.Polytope:
                        Kv = pc.Region(list_poly=[Kv])

                    if not cached_initset.list_poly:
                        contain = False
                    else:
                        contain = True
                        for poly in cached_initset.list_poly:
                            for new_poly in Kv.list_poly:
                                poly_box = pc.bounding_box(poly)
                                new_poly_box = pc.bounding_box(new_poly)
                                if not PolyUtils.does_rect_contain(poly_box, new_poly_box):
                                    contain = False
                    if contain:
                        print("One fixed point reached")
                        curfix = True
                        cursafe = False
                        curreftree = {}
                    else:
                        Av_p, curreftree, curfix, cursafe, full_tube = \
                            self.SymAR(Av_p, pv_p, pv_s, Kv, Rv_p, refine_depth, traversal_parent, full_tube, depth + 1)
                    if not curfix:
                        fixed_point = False
                    for key in curreftree:
                        if key in reftree:
                            for element in curreftree[key]:
                                if element not in reftree[key]:
                                    reftree[key].append(element)
                        else:
                            reftree[key] = curreftree[key]
                return Av_p, reftree, True, False, full_tube
        return Av_p, reftree, fixed_point, True, full_tube

    def compute_agent_full_tube_dfs(self, cur_agent: Agent, t = 0):
        full_tube: List[ReachtubeSegment] = []
        look_back: List[List[int]] = [[0, 0]]
        # list of pairs of index of a jump, look back number of indices
        # all polytopes in the tube between two consecutive indices would have a lookback equal to the previous one
        time_passed = 0
        visited = {}
        traversal_parent = []
        mode_id = cur_agent.mode_to_abs_mode[cur_agent.initial_mode_id]
        depth = 0
        refine_depth = 0

        transform_information = cur_agent.get_transform_information(cur_agent.mode_list[cur_agent.initial_mode_id])
        root_initset: pc.Region = cur_agent.transform_poly_to_virtual(cur_agent.initial_set, transform_information)
        cur_agent, reftree, fixedpoint, safe, full_tube = self.SymAR(cur_agent, mode_id, None, root_initset, None, refine_depth, traversal_parent, full_tube, 0)

        print("execution time before transform back to original tube : ", (time.time() - t) / 60.0)
        # print("Total number of refinements: ", refine_counter)
        full_tube = []
        full_trace = []
        traversal_queue = deque([cur_agent.initial_mode_id])
        while len(traversal_queue) > 0:
            mode_id = traversal_queue.popleft()
            abs_reachset = cur_agent.abs_mode_reachset[cur_agent.mode_to_abs_mode[mode_id]]
            # abs_trace = cur_agent.abs_mode_trace[cur_agent.mode_to_abs_mode[mode_id]]
            if len(abs_reachset) != 0:
                transform_information = cur_agent.get_transform_information(
                    cur_agent.mode_list[mode_id])
                tube_list = []
                rect_tube_list = []
                trace_list = []
                for tube in abs_reachset:
                    tube_list.append([cur_agent.transform_poly_from_virtual(pc.box2poly(tube[i][:, :].T),
                                                                            transform_information)
                                      for i in range(len(tube))])
                    rect_tube_list.append([])
                    for poly in tube_list[-1]:
                        rect_tube_list.append(PolyUtils.get_bounding_box(poly))

                # for trace in abs_trace:
                #     trace_list.append(cur_agent.transform_trace_from_virtual(trace, transform_information))
                reachset_segment = ReachtubeSegment(tube_list, rect_tube_list, trace_list, -1,
                                                    -1, np.array([]), cur_agent.mode_to_abs_mode[mode_id])
                cur_agent.number_of_segments_transformed += 1
                cur_agent.number_of_tubes_transformed += len(reachset_segment.tube_list)
                full_tube.append(reachset_segment)
                full_trace.append(trace_list)
                for ind, edge_id in enumerate(cur_agent.mode_neighbors[mode_id]):
                    traversal_queue.extend([cur_agent.edge_list[edge_id].dest])
            else:
                print("abs_reachset empty", mode_id, cur_agent.mode_to_abs_mode[mode_id])
                # pdb.set_trace()
        self.agents_list[0] = cur_agent
        self.agents_list[0].is_safe = safe
        print(f"full tube length {len(full_tube)}")
        simulation_trace = []
        # for i in range(10):
        #     pt = cur_agent.sample_initset()
        #     trace = self.hybridSimulation(cur_agent,pt)
        #     simulation_trace.append(trace)
        return full_tube, simulation_trace, look_back

    def simulate(self, waypoint, initset):
        dynamics_path = "multvr/examples/NNquadrotor"
        sys.path.append(os.path.abspath(dynamics_path))
        mod_name = dynamics_path.replace('/','.')
        module = importlib.import_module(mod_name)
        sys.path.pop()
        dryvr_mode = str(waypoint.mode_parameters)
        dryvr_mode = dryvr_mode.replace(",", ";")

        TC_Simulate = module.TC_Simulate
        res = TC_Simulate(dryvr_mode, initset, waypoint.time_bound)
        res = res[:,1:]
        return res

    def hybridSimulation(self, cur_agent: Agent, initial_state):
        trace = []
        root_mode_id = cur_agent.initial_mode_id
        traversal_queue = deque([(root_mode_id, initial_state)])
        while len(traversal_queue) > 0:
            tmp = traversal_queue.popleft()
            mode_id = tmp[0]
            mode_initset = tmp[1]
            waypoint: Waypoint = cur_agent.mode_list[mode_id]
            trace_segment = self.simulate(waypoint, mode_initset)
            truncate_idx = -1
            for edge_ind in cur_agent.mode_neighbors[mode_id]:
                guard = cur_agent.edge_list[edge_ind].guard
                guard = pc.box2poly(guard.T)
                inter = guard.contains(trace_segment.T)
                inter_ind = np.where(inter == True)[0]
                if len(inter_ind) > 0:
                    next_init_id = np.random.choice(inter_ind)
                    next_initset = trace_segment[next_init_id,:]
                    next_mode_id = cur_agent.edge_list[edge_ind].dest
                    traversal_queue.append((next_mode_id, next_initset))
                    if next_init_id > truncate_idx:
                        truncate_idx = next_init_id
            trace.append(trace_segment[:truncate_idx+1,:])
        return trace

    def verify(self, dynamic_safety=True, use_dryvr=False):
        Nref = 0
        S = 0
        S_vi = 0
        E_vi = 0
        S_vf = 0
        E_vf = 0
        Rc = 0
        Rt = 0
        Tt = 0

        # TODO make lookback a numpy array instead
        if use_dryvr:
            dryvr_fixed_point = False
            dryvr_time_bound = 4
            # self.max_dryvr_time_horizon = 8
            while dryvr_time_bound <= self.max_dryvr_time_horizon and not dryvr_fixed_point:
                self.dryvr_path = "/Users/husseinsibai/Desktop/DryVR_0.2/"  # TODO: change to be an input
                self.dryvrmode_to_virtualmode = {}
                self.construct_abs_dryvr_input_file(dryvr_time_bound)
                self.run_dryvr()
                self.parse_dryvr_output()
                dryvr_fixed_point = True
                # check for fixed point
                for abs_prev_waypoint in self.abs_reach:
                    for abs_cur_waypoint in self.abs_reach:
                        abs_cur_edge = tuple([abs_prev_waypoint, abs_cur_waypoint])
                        print("abs_cur_edge: ", abs_cur_edge)
                        if abs_cur_edge in self.abs_edges_guards:
                            print("it is an edge")
                            _, _, guards_inter_list = TubeMaster.intersect_waypoint_list(
                                self.abs_reach[abs_prev_waypoint],
                                self.abs_edges_guards[abs_cur_edge])
                            print("guards_inter_list: ", guards_inter_list)
                            for rect in guards_inter_list:
                                possible_next_initset_rect = PolyUtils.get_bounding_box(
                                    self.agents_list[-1].transform_poly_to_virtual(
                                        self.agents_list[-1].transform_poly_from_virtual(
                                            pc.box2poly(rect[0].T),
                                            rect[1]), rect[2]))
                                print("abs_cur_waypoint: ", abs_cur_waypoint)
                                print("possible_next_initset_rect:", possible_next_initset_rect)
                                if len(self.abs_reach[abs_cur_waypoint]) > 0:
                                    print("self.abs_reach[abs_cur_waypoint][0]:", self.abs_reach[abs_cur_waypoint][0])
                                    if not PolyUtils.does_rect_contain(possible_next_initset_rect,
                                                                       self.abs_reach[abs_cur_waypoint][0]):
                                        print("INITIAL SET NOT COMPUTED YET :/")
                                        dryvr_fixed_point = False
                                        break
                                else:
                                    dryvr_fixed_point = False
                                    break

                            if not dryvr_fixed_point:
                                break
                        if not dryvr_fixed_point:
                            break
                    if not dryvr_fixed_point:
                        break
                if dryvr_fixed_point:
                    break
                else:
                    dryvr_time_bound = dryvr_time_bound * 2
                print("dryvr_time_bound: ", dryvr_time_bound)

            if dryvr_fixed_point:
                print("FIXED POINT HAS BEEN REACHED")
            self.tube_master.reached_fixed_point = True
        t = time.time()
        agents_look_back: List[List[List[int]]] = []
        max_tube_length = 0
        prev_computed = 0
        prev_saved = 0
        prev_ultrasaved = 0
        for i in range(self.num_agents):
            agent = self.agents_list[i]
            # only checks static safety
            agent_tubesegments: List[ReachtubeSegment]
            agent_look_back: List[List[int]]
            if self.sym_level == 0:
                agent_tubesegments, agent_traces, agent_look_back = self.compute_agent_full_tube(agent, t)
            else:
                # agent_tubesegments, agent_traces, agent_look_back = self.compute_agent_full_tube(agent, t)
                # agent_tubesegments, agent_traces, agent_look_back = self.compute_agent_full_tube_bfs(agent, t)
                agent_tubesegments, agent_traces, agent_look_back = self.compute_agent_full_tube_dfs(agent, t)
            agent = self.agents_list[i]
            for key in agent.abs_mode_reachset:
                print(f"abs_mode {key} len {len(agent.abs_mode_reachset[key])}")
            print("initset volume: ")
            volume_list = []
            for agent_tubesegment in agent_tubesegments:
                curr_initset_volume = 0
                for tube in agent_tubesegment.tube_list:
                    curr_initset_volume += PolyUtils.get_rect_volume(PolyUtils.get_bounding_box(tube[0]))
                volume_list.append(
                    PolyUtils.get_rect_volume(PolyUtils.get_bounding_box(agent_tubesegment.tube_list[-1][0])))
                # print("Is segment safe?", self.is_tube_static_safe(agent_tubesegment, self.unsafe_set))
            print(volume_list)
            if agent_look_back[-1][0] > max_tube_length:
                max_tube_length = agent_look_back[-1][0]
            self.agents_full_tubes.append(agent_tubesegments)
            self.agents_full_trace.append(agent_traces)
            agents_look_back.append(agent_look_back)
            print("finished agent #", len(self.agents_full_tubes))
            print("Nref", agent.num_refinements)
            model = self.agent_dynamics_ids[i]
            # print("Number of computed tubes: ", self.tube_master.tube_tools[model].computed_counter - prev_computed)
            # print("Number of saved tubes: ", self.tube_master.tube_tools[model].saved_counter - prev_saved)
            print("|S|: ", len(agent.mode_list))
            print("|S_v|^i: ", agent.initial_abs_mode_num)
            print("|E_v|^i: ", agent.initial_abs_edge_num)
            print("|S_v|^f: ", len(agent.abs_mode_list))
            print("|E_v|^f: ", len(agent.abs_edge_list))

            Nref = agent.num_refinements
            S = len(agent.mode_list)
            S_vi = agent.initial_abs_mode_num
            E_vi = agent.initial_abs_edge_num
            S_vf = len(agent.abs_mode_list)
            E_vf = len(agent.abs_edge_list)

            # prev_computed = self.tube_master.tube_tools[model].computed_counter
            # prev_saved = self.tube_master.tube_tools[model].saved_counter

        # pdb.set_trace()
        if dynamic_safety:
            agents_segment_idx = [0] * self.num_agents
            agents_segment_entry_idx = [0] * self.num_agents
            for i in range(max_tube_length):
                for d1 in range(self.num_agents):  # this would not be true once we have agents of different types
                    # if i > agents_look_back[d1][-1][0]:
                    #   continue
                    if agents_segment_idx[d1] == -1:
                        continue
                    # if i > agents_look_back[d1][agents_segment_idx[d1]][0]:
                    #    agents_segment_idx[d1] += 1
                    #    agents_segment_entry_idx[d1] = 0
                    d1_cur_look_back = agents_look_back[d1][agents_segment_idx[d1]][1]
                    for d2 in range(d1):
                        min_d2_rel_idx = i - d1_cur_look_back
                        if i - d1_cur_look_back >= agents_look_back[d2][agents_segment_idx[d2]][0]:
                            continue
                        d2_cur_segment_idx = agents_segment_idx[d2]
                        d2_cur_segment_entry_idx = agents_segment_entry_idx[d2]
                        if d2_cur_segment_idx == -1:
                            j = i
                            while j >= agents_look_back[d2][-1][0]:
                                d1_cur_look_back -= 1
                                j -= 1
                            d2_cur_segment_idx = len(self.agents_full_tubes[d2]) - 1
                            d2_cur_segment_entry_idx = len(self.agents_full_tubes[d2][d2_cur_segment_idx].tube) - 1

                        d2_num_steps_back = d1_cur_look_back

                        #    d1_cur_look_back -= 1
                        # if j >= agents_look_back[d2][agents_segment_idx[d2]][0] and agents_segment_idx[d2] < len(self.agents_full_tubes[d2]) - 1:
                        #    agents_segment_idx[d2] += 1
                        # else:
                        #    j = i
                        # maximization with zero here is just for safety but otherwise it shouldn't be needed
                        while d2_num_steps_back > 0:
                            if d2_cur_segment_entry_idx < d2_num_steps_back:
                                if d2_cur_segment_idx > 0:
                                    d2_num_steps_back -= d2_cur_segment_entry_idx
                                    d2_cur_segment_idx -= 1
                                    d2_cur_segment_entry_idx = len(self.agents_full_tubes[d2][d2_cur_segment_idx].tube) \
                                                               - 1
                                else:
                                    d2_num_steps_back = 0
                                    d2_cur_segment_entry_idx = 0
                                    d2_cur_segment_idx = 0
                            else:
                                d2_cur_segment_entry_idx -= d2_num_steps_back
                                d2_num_steps_back = 0
                        counter = 0
                        d2_look_back_includes_i = False
                        # d2_cur_segment_entry_idx + \
                        #                             agents_look_back[d2][d2_cur_segment_idx][0] - \
                        #                             agents_look_back[d2][d2_cur_segment_idx][1] <= i
                        while counter < d1_cur_look_back or d2_look_back_includes_i:
                            # check the intersection of full_tube[d1][i] and full_tube[d2][j]
                            # if not pc.is_empty(pc.intersect(
                            if not PolyUtils.is_polytope_intersection_empty(
                                    self.agents_full_tubes[d1][agents_segment_idx[d1]].tube[
                                        agents_segment_entry_idx[d1]],
                                    self.agents_full_tubes[d2][d2_cur_segment_idx].tube[d2_cur_segment_entry_idx]):
                                print("The system is unsafe since drone ", d1, " intersects with drone ", d2,
                                      "at time ", i
                                      * self.time_step)
                                # return False
                            d2_cur_segment_entry_idx += 1
                            if d2_cur_segment_entry_idx >= len(self.agents_full_tubes[d2][d2_cur_segment_idx].tube):
                                d2_cur_segment_idx += 1
                                if d2_cur_segment_idx >= len(self.agents_full_tubes[d2]):
                                    break
                                d2_cur_segment_entry_idx = 0
                            d2_look_back_includes_i = d2_cur_segment_entry_idx + \
                                                      agents_look_back[d2][d2_cur_segment_idx][0] \
                                                      - agents_look_back[d2][d2_cur_segment_idx][1] <= i
                            counter += 1
                    for d in range(self.num_agents):
                        if agents_segment_idx[d] > -1:
                            agents_segment_entry_idx[d] += 1
                            if agents_segment_entry_idx[d] >= len(
                                    self.agents_full_tubes[d][agents_segment_idx[d]].tube_list[-1]):
                                agents_segment_idx[d] += 1
                                agents_segment_entry_idx[d] = 0
                            if agents_segment_idx[d] >= len(self.agents_full_tubes[d]):
                                agents_segment_idx[d] = -1

        # print("agents look back: ", agents_look_back)
        # print("max_tube_length: ", max_tube_length)
        for model in set(self.agent_dynamics_ids):
            print("Rc: ", self.tube_master.tube_tools[model].computed_counter)
            Rc = self.tube_master.tube_tools[model].computed_counter
        print("Rt: ", self.agents_list[-1].reachtube_time/60)
        print("Tt: ", (time.time() - t) / 60.0)

        Rt = self.agents_list[-1].reachtube_time/60
        Tt = (time.time() - t) / 60.0
        # print("transform time per agent:", [agent.transform_time / 60.0 for agent in self.agents_list])

        sys.stdout = self.old_stdout
        
        print("###################")
        print("###################")
        print("Nref", Nref)
        print("|S|: ", S)
        print("|S_v|^i: ", S_vi)
        print("|E_v|^i: ", E_vi)
        print("|S_v|^f: ", S_vf)
        print("|E_v|^f: ", E_vf)
        print("Rc: ", Rc)
        print("Rt: ", Rt)
        print("Tt: ", Tt)
        
        return self.agents_list[-1].is_safe, [Nref, S, S_vi, E_vi, S_vf, E_vf, Rc, Rt, Tt]

    def plot(self):
        is_unified = False
        if self.tube_master.sym_level == 2:
            abs_reach = self.agents_list[
               -1].abs_mode_reachset  # self.tube_master.tube_tools[self.agents_list[-1].dynamics].abs_reach
            Plotter.plot(self.num_agents, self.agents_full_tubes, self.agents_full_trace, self.agents_list, self.global_unsafe_set, self.initsets, self.goalsets,
                         self.sym_level,
                         plot_step=1)
            Plotter.plot_virtual(self.agents_list[-1].abs_mode_reachset, self.agents_list,
                                 self.agents_list[-1].abs_mode_neighbors, plot_step=1)
        else:
            Plotter.plot(self.num_agents, self.agents_full_tubes,  self.agents_full_trace, self.agents_list, self.global_unsafe_set, self.initsets, self.goalsets,
                         self.sym_level,
                         plot_step=1)
        pass


def main():
    # scene = VerificationScenario(str(os.path.abspath(
    #    './../examples/scenarios/Linear3D_S_trans_2D.json')), "Linear3D_S_trans_2D.json")  # Linear3D_S_trans_2D # Drone3D_S.json # ThreeFixwingedDrones_3d # snowflake.json
    # result = scene.verify(False)
    now = str(datetime.datetime.now())
    now = now.replace(' ','_')
    now = now.replace(':','-')
    now = now.split('.')[0]
    log_fn = f"./log/log_{now}"
    log_file_name = log_fn
    log_file = open(log_fn,'w+')
    scene = VerificationScenario(str(os.path.abspath(sys.argv[1])), log_file=log_file, seed = True)
    result,_ = scene.verify(use_dryvr=False)
    log_file.close()
    # TODO support return of unsafe result.
    print("Is system safe?", result)
    """
    if type(result) == bool and result:
        print("System is safe")
    else:
        print("System safety is unknown")
    """
    scene.plot()
    # trace = scene.hybridSimulation(scene.agents_list[0],[5.5, 5.5, 0, 0, 0, 0])
    # for segment in trace:
    #     plt.plot(segment[:,0], segment[:,1], 'b')
    # plt.show()

if __name__ == '__main__':
    # main()
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # pr.print_stats()
