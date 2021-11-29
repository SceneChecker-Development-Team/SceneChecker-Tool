import polytope as pc
from src.Waypoint import Waypoint
from src.Edge import Edge
import numpy as np
from scipy.integrate import odeint
import math
from src.ReachtubeSegment import ReachtubeSegment
from src.PolyUtils import PolyUtils
from typing import List, Set, Tuple, Dict, Optional
import matplotlib
import yaml
import xml.etree.ElementTree as ET
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from src.DiscrepancyLearning import DiscrepancyLearning
from src.ParseModel import importFunctions
import copy
import subprocess
import pdb
import ast
import time
import re
from src.DryVRUtils import DryVRUtils
DRONE_TYPE = 1

import random

class Agent:

    def __init__(self, variables_list: List[str], mode_list: List[Waypoint], edge_list: List[Edge],
                 mode_neighbors: Dict[Waypoint, List[Waypoint]], mode_parents: Dict[Waypoint, List[int]],
                 initial_set: pc.Polytope, initial_mode_id: int, dynamics: str, function_arr, abs_mode_list: List[Waypoint],
                 abs_edge_list: List[Edge],
                 abs_mode_neighbors: Dict[int, List[int]],
                 abs_mode_parents: Dict[Waypoint, List[int]],
                 abs_initial_set: pc.Polytope, mode_to_abs_mode: Dict[int, int],
                 abs_mode_to_mode: Dict[int, List[int]]) -> None:
        self.variables_list = variables_list
        self.mode_list: List[Waypoint] = mode_list
        self.edge_list: List[Edge] = edge_list
        self.mode_neighbors: Dict[Waypoint, List[Waypoint]] = mode_neighbors
        self.mode_parents: Dict[Waypoint, List[Waypoint]] = mode_parents
        self.initial_set: pc.Polytope = initial_set
        self.initial_mode_id = initial_mode_id
        self.mode_initset: Dict[Waypoint, pc.Region] = {}
        self.mode_reachset: Dict[Waypoint, List[np.array]] = {}
        self.mode_trace: Dict[Waypoint, List[np.array]] = {}
        # self.local_unsafe_sets: pc.Region = local_unsafe_sets
        self.dynamics: str = dynamics
        self.transform_time = 0
        self.dim = initial_set.dim
        self.TC_Simulate = function_arr[0]
        self.get_transform_information = function_arr[1]
        self.transform_poly_to_virtual = function_arr[2]
        self.transform_mode_to_virtual = function_arr[3]
        self.transform_poly_from_virtual = function_arr[4]
        self.transform_trace_from_virtual = function_arr[10]
        self.transform_mode_from_virtual = function_arr[5]
        self.get_virtual_mode_parameters = function_arr[6]
        self.transform_state_from_then_to_virtual_dryvr_string = function_arr[7]
        self.get_flowstar_parameters = function_arr[8]
        self.get_sherlock_parameters = function_arr[9]

        self.initial_abs_mode_num = len(abs_mode_list)
        self.initial_abs_edge_num = len(abs_edge_list)
        self.abs_mode_list: List[Waypoint] = abs_mode_list
        self.abs_edge_list: List[Edge] = abs_edge_list
        self.abs_mode_neighbors: Dict[int, List[int]] = abs_mode_neighbors
        self.abs_mode_parents: Dict[int, List[int]] = abs_mode_parents

        self.mode_to_abs_mode: Dict[int, int] = mode_to_abs_mode
        self.abs_mode_to_mode: Dict[int, List[int]] = abs_mode_to_mode

        self.number_of_segments_computed = 0
        self.number_of_segments_transformed = 0
        self.number_of_tubes_computed = 0
        self.number_of_tubes_transformed = 0
        self.abs_per_virtual_mode_tube_counter: Dict[Tuple[List[float]], int] = {}
        self.reached_fixed_point = False

        self.abs_mode_initset: Dict[int, pc.Region] = {}
        self.abs_mode_reachset: Dict[int, List[np.array]] = {}
        self.abs_mode_trace: Dict[int, List[np.array]] = {}

        self.initset_union: pc.Region = pc.Region(list_poly=[])
        self.initset_union_rects: List = []
        self.uncovered_sets = pc.Region(list_poly=[])
        self.uncovered_sets_per_mode = dict()
        self.uncovered_rects_per_mode = dict()
        self.abs_initset_rects = {}
        self.clean_caches()
        self.is_safe: bool = True

        self.num_refinements:int = 0
        self.max_refinements:int = 0
        self.reachtube_time = 0

        self.max_depth = Agent.get_max_depth(self.edge_list,self.initial_mode_id)

        self.seed = 0


    @staticmethod
    def find_edge_from_mode(edge_list, src_idx):
        idx_list = []
        for idx in range(len(edge_list)):
            if edge_list[idx].source == src_idx:
                idx_list.append(idx)
        return idx_list

    @staticmethod
    def get_max_depth(edge_list, init_mode_id):
        max_depth = 0
        traversal_queue = [(init_mode_id, 0)]
        while traversal_queue:
            node = traversal_queue.pop(0)
            mode = node[0]
            depth = node[1]
            if depth > max_depth:
                max_depth = depth
            neighbors = Agent.find_edge_from_mode(edge_list, mode)
            for idx in neighbors:
                edge = edge_list[idx]
                traversal_queue.append((edge.dest, depth+1))
        return max_depth

    @staticmethod
    def time_list(time_step: float, time_bound: float) -> np.array:
        return np.arange(0, time_bound + time_step, time_step)

    @staticmethod
    def reached_guard(s: np.array, edge: Edge):
        guard: np.array = edge.guard
        # if s[0] + start_time > waypoint.time_bound:
        #    return False
        return np.all(guard[0, :] <= s[1:]) and np.all(guard[1, :] >= s[1:])

    def clean_caches(self):
        for abs_mode in self.abs_mode_list:
            self.abs_initset_rects[abs_mode.id] = []
            self.abs_per_virtual_mode_tube_counter[abs_mode.id] = 0
            self.uncovered_sets_per_mode[abs_mode.id] = pc.Region(list_poly=[])
            self.uncovered_rects_per_mode[abs_mode.id] = []
            self.abs_mode_initset[abs_mode.id] = pc.Region(list_poly=[])
            self.abs_mode_reachset[abs_mode.id] = []

    def create_caches(self, abs_mode_id):
        self.abs_initset_rects[abs_mode_id] = []
        self.abs_per_virtual_mode_tube_counter[abs_mode_id] = 0
        self.uncovered_sets_per_mode[abs_mode_id] = pc.Region(list_poly=[])
        self.uncovered_rects_per_mode[abs_mode_id] = []
        self.abs_mode_initset[abs_mode_id] = pc.Region(list_poly=[])
        self.abs_mode_reachset[abs_mode_id] = []

    def clean_mode_caches(self, abs_mode_id):
        self.uncovered_sets_per_mode[abs_mode_id] = pc.Region(list_poly=[])
        self.abs_mode_initset[abs_mode_id] = pc.Region(list_poly=[])
        self.abs_mode_reachset[abs_mode_id] = []

    def check_fixed_point(self):
        # Checking for fixed point!!
        fixed_point = True
        for ind, abs_mode_ind in enumerate(self.abs_mode_initset):
            '''
            if not abs_mode_ind in self.uncovered_sets_per_mode:
                self.uncovered_sets_per_mode[abs_mode_ind] = pc.Region(list_poly=[])
            if not abs_mode_ind in self.uncovered_rects_per_mode:
                self.uncovered_rects_per_mode[abs_mode_ind] = []
            '''
            '''
            new_uncovered_sets = pc.Region(list_poly=[])
            if type(self.uncovered_sets_per_mode[abs_mode_ind]) == pc.Polytope:
                self.uncovered_sets_per_mode[abs_mode_ind] = pc.Region(
                    list_poly=[self.uncovered_sets_per_mode[abs_mode_ind]])

            indices_to_delete = []

            for i in range(len(self.uncovered_rects_per_mode[abs_mode_ind])):
                for init_rect in self.abs_initset_rects[abs_mode_ind]:
                    if PolyUtils.does_rect_contain(self.uncovered_rects_per_mode[abs_mode_ind][i], init_rect):
                        indices_to_delete.append(i)
                        break

            for index in sorted(indices_to_delete, reverse=True):
                self.uncovered_rects_per_mode[abs_mode_ind].pop(index)

            for i in range(len(self.uncovered_sets_per_mode[abs_mode_ind].list_poly)):
                if not pc.is_subset(self.uncovered_sets_per_mode[abs_mode_ind].list_poly[i],
                                    self.abs_mode_initset[abs_mode_ind]):
                    new_uncovered_sets = pc.union(new_uncovered_sets,
                                                  self.uncovered_sets_per_mode[abs_mode_ind].list_poly[i])
            '''
            if type(self.uncovered_sets_per_mode[abs_mode_ind]) == pc.Polytope:
                self.uncovered_sets_per_mode[abs_mode_ind] = pc.Region(list_poly=[self.uncovered_sets_per_mode[abs_mode_ind]])
            print("len(self.uncovered_sets.list_poly), ", ind, "th", " mode ", abs_mode_ind, " is: ",
                  len(self.uncovered_sets_per_mode[abs_mode_ind].list_poly))
            if (not pc.is_empty(self.uncovered_sets_per_mode[abs_mode_ind])):
                # len(self.uncovered_sets_per_mode[abs_mode_ind].list_poly) > 0  # not pc.is_empty(new_uncovered_sets) and
                fixed_point = False
            # self.uncovered_sets_per_mode[abs_mode_ind] = new_uncovered_sets

        if fixed_point:
            print("FIXED POINT HAS BEEN REACHED!!!")
        '''
        if pc.is_empty(self.uncovered_sets):
            print("NO MORE CACHE MISSES")
        '''
        return fixed_point

    def decompose_abs_mode(self, abs_mode_id, unsafeset_inter_list):
        abs_mode = self.abs_mode_list[abs_mode_id]
        self.abs_mode_list.append(Waypoint(abs_mode.mode, abs_mode.mode_parameters, abs_mode.time_bound,
                                           len(self.abs_mode_list)))
        abs_mode_id_2 = len(self.abs_mode_list) - 1
        self.create_caches(abs_mode_id_2)
        orig_mode_list = self.abs_mode_to_mode[abs_mode_id]
        print("unsafeset_inter_list: ", unsafeset_inter_list)
        print("orig_mode_list: ", orig_mode_list)
        if False: # len(unsafeset_inter_list) != 0 and len(orig_mode_list) > len(unsafeset_inter_list):
            orig_mode_list_1 = []
            orig_mode_list_2 = []
            for orig_mode_id in orig_mode_list:
                if orig_mode_id in unsafeset_inter_list:
                    orig_mode_list_2.append(orig_mode_id)
                else:
                    orig_mode_list_1.append(orig_mode_id)
        else:
            orig_mode_list_1 = orig_mode_list[:math.floor(len(orig_mode_list)/2)]
            orig_mode_list_2 = orig_mode_list[math.floor(len(orig_mode_list)/2):]
        # print("Modes ", orig_mode_list_1, " going to abstract mode ", abs_mode_id)
        # print("Modes ", orig_mode_list_2, " going to abstract mode ", abs_mode_id_2)
        unsafe_set_list_1 = []
        unsafe_set_list_2 = []
        for unsafeset in abs_mode.unsafeset_list:
            if unsafeset.mode_id in orig_mode_list_1:
                unsafe_set_list_1.append(unsafeset)
            elif unsafeset.mode_id in orig_mode_list_2:
                unsafe_set_list_2.append(unsafeset)
        self.abs_mode_list[abs_mode_id].unsafeset_list = unsafe_set_list_1
        self.abs_mode_list[-1].unsafeset_list = unsafe_set_list_2
        self.abs_mode_to_mode[abs_mode_id] = orig_mode_list_1
        self.abs_mode_to_mode[abs_mode_id_2] = orig_mode_list_2
        for mode_id in orig_mode_list_2:
            self.mode_to_abs_mode[mode_id] = abs_mode_id_2
        self.abs_mode_parents[abs_mode_id_2] = []
        self.abs_mode_neighbors[abs_mode_id_2] = []

        ######### splitting parents edges ##############
        for abs_edge_id in list(self.abs_mode_parents[abs_mode_id]):
            abs_edge = self.abs_edge_list[abs_edge_id]
            #if abs_edge.source == 3 and abs_edge.dest == 1:
            #    #pdb.set_trace()
            if abs_edge.source == abs_edge.dest:
                # print("skipping abstract edge [", abs_edge.source, ", ", abs_edge.dest, "]")
                continue
            abs_guard_1 = []
            abs_guard_2 = []
            abs_guard_region_1 = pc.Region(list_poly=[])
            abs_guard_region_2 = pc.Region(list_poly=[])
            for edge_info in abs_edge.guard:
                if self.edge_list[edge_info[3]].dest in orig_mode_list_1:
                    abs_guard_1.append(edge_info)
                    #print("going to the original parent edge [", self.edge_list[edge_info[3]].source,
                    #      ", ", self.edge_list[edge_info[3]].dest, "]")
                    abs_guard_region_1 = PolyUtils.get_region_union(abs_guard_region_1, pc.box2poly(edge_info[0].T))
                elif self.edge_list[edge_info[3]].dest in orig_mode_list_2:
                    abs_guard_2.append(edge_info)
                    #print("going to the new parent edge [", self.edge_list[edge_info[3]].source,
                    #      ", ", self.edge_list[edge_info[3]].dest, "]")
                    abs_guard_region_2 = PolyUtils.get_region_union(abs_guard_region_2, pc.box2poly(edge_info[0].T))
                else:
                    pdb.set_trace()

            if len(abs_guard_1) + len(abs_guard_2) != len(abs_edge.guard):
                pdb.set_trace()

            self.abs_edge_list[abs_edge_id].guard = abs_guard_1
            self.abs_edge_list[abs_edge_id].region_guard = abs_guard_region_1
            if len(abs_guard_1) == 0:
                ## delete edge if it's no longer useful
                if not abs_edge_id in self.abs_mode_parents[abs_mode_id]:
                    pdb.set_trace()
                self.abs_mode_parents[abs_mode_id].remove(abs_edge_id)
                if not abs_edge_id in self.abs_mode_neighbors[abs_edge.source]:
                    pdb.set_trace()
                self.abs_mode_neighbors[abs_edge.source].remove(abs_edge_id)
                # print("removed abstract edge [", abs_edge.source, ", ", abs_edge.dest, "]")
            if len(abs_guard_2) > 0:
                self.abs_edge_list.append(Edge(abs_edge.source, abs_mode_id_2,
                                               len(self.abs_edge_list), abs_guard_2, abs_guard_region_2))
                self.abs_mode_parents[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])
                self.abs_mode_neighbors[abs_edge.source].extend([len(self.abs_edge_list) - 1])

        if len(self.abs_edge_list)>=48:
            print("Stop here")
        """
        for abs_edge_id in self.abs_mode_parents[abs_mode_id]:
            abs_edge = self.abs_edge_list[abs_edge_id]
            print("######abs_mode_id: The edges of the abstract edge: [", abs_edge.source, ", ", abs_edge.dest, "] are: ")
            for edge_info in abs_edge.guard:
                print("[", self.edge_list[edge_info[3]].source,
                      ", ", self.edge_list[edge_info[3]].dest, "]")
        for abs_edge_id in self.abs_mode_parents[abs_mode_id_2]:
            abs_edge = self.abs_edge_list[abs_edge_id]
            print("######abs_mode_id_2:  The edges of the abstract edge: [", abs_edge.source, ", ", abs_edge.dest, "] are: ")
            for edge_info in abs_edge.guard:
                print("[", self.edge_list[edge_info[3]].source,
                      ", ", self.edge_list[edge_info[3]].dest, "]")
        """


        ######### splitting children edges #############
        for abs_edge_id in list(self.abs_mode_neighbors[abs_mode_id]):
            abs_edge = self.abs_edge_list[abs_edge_id]
            if abs_edge.source == abs_edge.dest:
                # print("skipping abstract edge [", abs_edge.source, ", ", abs_edge.dest, "]")
                continue
            abs_guard_1 = []
            abs_guard_2 = []
            abs_guard_region_1 = pc.Region(list_poly=[])
            abs_guard_region_2 = pc.Region(list_poly=[])
            for edge_info in abs_edge.guard:
                if self.edge_list[edge_info[3]].source in orig_mode_list_1:
                    abs_guard_1.append(edge_info)
                    #print("going to the original child edge [", self.edge_list[edge_info[3]].source,
                    #      ", ", self.edge_list[edge_info[3]].dest, "]")
                    abs_guard_region_1 = PolyUtils.get_region_union(abs_guard_region_1, pc.box2poly(edge_info[0].T))
                elif self.edge_list[edge_info[3]].source in orig_mode_list_2:
                    abs_guard_2.append(edge_info)
                    # print("going to the new child edge [", self.edge_list[edge_info[3]].source,
                    #      ", ", self.edge_list[edge_info[3]].dest, "]")
                    abs_guard_region_2 = PolyUtils.get_region_union(abs_guard_region_2, pc.box2poly(edge_info[0].T))
                else:
                    pdb.set_trace()

            if len(abs_guard_1) + len(abs_guard_2) != len(abs_edge.guard):
                pdb.set_trace()

            self.abs_edge_list[abs_edge_id].guard = abs_guard_1
            self.abs_edge_list[abs_edge_id].region_guard = abs_guard_region_1
            if len(abs_guard_1) == 0:
                ## delete edge if it's no longer useful
                if not abs_edge_id in self.abs_mode_parents[abs_edge.dest]:
                    pdb.set_trace()
                self.abs_mode_parents[abs_edge.dest].remove(abs_edge_id)
                if not abs_edge_id in self.abs_mode_neighbors[abs_mode_id]:
                    pdb.set_trace()
                self.abs_mode_neighbors[abs_mode_id].remove(abs_edge_id)
            if len(abs_guard_2) > 0:
                self.abs_edge_list.append(Edge(abs_mode_id_2, abs_edge.dest,
                                               len(self.abs_edge_list), abs_guard_2, abs_guard_region_2))
                self.abs_mode_parents[abs_edge.dest].extend([len(self.abs_edge_list) - 1])
                self.abs_mode_neighbors[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])

        ########## handling self edges ###################
        for abs_edge_id in list(self.abs_mode_neighbors[abs_mode_id]):
            abs_edge = self.abs_edge_list[abs_edge_id]
            if abs_edge.source == abs_edge.dest:
                abs_guard_11 = []
                abs_guard_12 = []
                abs_guard_21 = []
                abs_guard_22 = []
                abs_guard_region_11 = pc.Region(list_poly=[])
                abs_guard_region_12 = pc.Region(list_poly=[])
                abs_guard_region_21 = pc.Region(list_poly=[])
                abs_guard_region_22 = pc.Region(list_poly=[])
                for edge_info in abs_edge.guard:
                    if (self.edge_list[edge_info[3]].source in orig_mode_list_1) and (self.edge_list[edge_info[3]].dest in orig_mode_list_1):
                        abs_guard_11.append(edge_info)
                        abs_guard_region_11 = PolyUtils.get_region_union(abs_guard_region_11, pc.box2poly(edge_info[0].T))
                    elif (self.edge_list[edge_info[3]].source in orig_mode_list_1) and (self.edge_list[edge_info[3]].dest in orig_mode_list_2):
                        abs_guard_12.append(edge_info)
                        abs_guard_region_12 = PolyUtils.get_region_union(abs_guard_region_12, pc.box2poly(edge_info[0].T))
                    elif (self.edge_list[edge_info[3]].source in orig_mode_list_2) and (self.edge_list[edge_info[3]].dest in orig_mode_list_1):
                        abs_guard_21.append(edge_info)
                        abs_guard_region_21 = PolyUtils.get_region_union(abs_guard_region_21, pc.box2poly(edge_info[0].T))
                    elif (self.edge_list[edge_info[3]].source in orig_mode_list_2) and (self.edge_list[edge_info[3]].dest in orig_mode_list_2):
                        abs_guard_22.append(edge_info)
                        abs_guard_region_22 = PolyUtils.get_region_union(abs_guard_region_22, pc.box2poly(edge_info[0].T))

                if len(abs_guard_11) + len(abs_guard_12) + len(abs_guard_21) + len(abs_guard_22) != len(abs_edge.guard):
                    pdb.set_trace()

                self.abs_edge_list[abs_edge_id].guard = abs_guard_11
                self.abs_edge_list[abs_edge_id].region_guard = abs_guard_region_11
                if len(abs_guard_11) == 0:
                    ## delete edge if it's no longer useful
                    if not abs_edge_id in self.abs_mode_parents[abs_mode_id]:
                        pdb.set_trace()
                    self.abs_mode_parents[abs_mode_id].remove(abs_edge_id)
                    if not abs_edge_id in self.abs_mode_neighbors[abs_mode_id]:
                        pdb.set_trace()
                    self.abs_mode_neighbors[abs_mode_id].remove(abs_edge_id)
                if len(abs_guard_12) > 0:
                    self.abs_edge_list.append(Edge(abs_mode_id, abs_mode_id_2,
                                                   len(self.abs_edge_list), abs_guard_12, abs_guard_region_12))
                    self.abs_mode_parents[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])
                    self.abs_mode_neighbors[abs_mode_id].extend([len(self.abs_edge_list) - 1])
                if len(abs_guard_21) > 0:
                    self.abs_edge_list.append(Edge(abs_mode_id_2, abs_mode_id,
                                                   len(self.abs_edge_list), abs_guard_21, abs_guard_region_21))
                    self.abs_mode_parents[abs_mode_id].extend([len(self.abs_edge_list) - 1])
                    self.abs_mode_neighbors[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])
                if len(abs_guard_22) > 0:
                    self.abs_edge_list.append(Edge(abs_mode_id_2, abs_mode_id_2,
                                                   len(self.abs_edge_list), abs_guard_22, abs_guard_region_22))
                    self.abs_mode_parents[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])
                    self.abs_mode_neighbors[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])

        ###### initializing caches for the new mode ##############
        # self.clean_caches()
        for abs_edge in self.abs_edge_list:
            # print("The edges of the abstract edge: [", abs_edge.source, ", ", abs_edge.dest, "] are: ")
            for edge_info in abs_edge.guard:
                # print("[", self.edge_list[edge_info[3]].source,
                #       ", ", self.edge_list[edge_info[3]].dest, "]")
                if not self.edge_list[edge_info[3]].source in self.abs_mode_to_mode[abs_edge.source]:
                    pdb.set_trace()
                if not self.edge_list[edge_info[3]].dest in self.abs_mode_to_mode[abs_edge.dest]:
                    pdb.set_trace()

    def decompose_abs_mode_nonReplace(self, abs_mode_id, unsafeset_inter_list):
        abs_mode = self.abs_mode_list[abs_mode_id]
        self.abs_mode_list.append(Waypoint(abs_mode.mode, abs_mode.mode_parameters, abs_mode.time_bound,
                                           len(self.abs_mode_list)))
        self.abs_mode_list.append(Waypoint(abs_mode.mode, abs_mode.mode_parameters, abs_mode.time_bound,
                                           len(self.abs_mode_list)))
        abs_mode_id_1 = len(self.abs_mode_list) - 2
        abs_mode_id_2 = len(self.abs_mode_list) - 1
        self.create_caches(abs_mode_id_1)
        self.create_caches(abs_mode_id_2)
        orig_mode_list = self.abs_mode_to_mode[abs_mode_id]
        print("unsafeset_inter_list: ", unsafeset_inter_list)
        print("orig_mode_list: ", orig_mode_list)
        if False: # len(unsafeset_inter_list) != 0 and len(orig_mode_list) > len(unsafeset_inter_list):
            orig_mode_list_1 = []
            orig_mode_list_2 = []
            for orig_mode_id in orig_mode_list:
                if orig_mode_id in unsafeset_inter_list:
                    orig_mode_list_2.append(orig_mode_id)
                else:
                    orig_mode_list_1.append(orig_mode_id)
        else:
            orig_mode_list_1 = orig_mode_list[:math.floor(len(orig_mode_list)/2)]
            orig_mode_list_2 = orig_mode_list[math.floor(len(orig_mode_list)/2):]
        # print("Modes ", orig_mode_list_1, " going to abstract mode ", abs_mode_id)
        # print("Modes ", orig_mode_list_2, " going to abstract mode ", abs_mode_id_2)
        unsafe_set_list_1 = []
        unsafe_set_list_2 = []
        for unsafeset in abs_mode.unsafeset_list:
            if unsafeset.mode_id in orig_mode_list_1:
                unsafe_set_list_1.append(unsafeset)
            elif unsafeset.mode_id in orig_mode_list_2:
                unsafe_set_list_2.append(unsafeset)
        self.abs_mode_list[abs_mode_id_1].unsafeset_list = unsafe_set_list_1
        self.abs_mode_list[abs_mode_id_2].unsafeset_list = unsafe_set_list_2
        self.abs_mode_to_mode[abs_mode_id_1] = orig_mode_list_1
        self.abs_mode_to_mode[abs_mode_id_2] = orig_mode_list_2
        for mode_id in orig_mode_list_1:
            self.mode_to_abs_mode[mode_id] = abs_mode_id_1
        for mode_id in orig_mode_list_2:
            self.mode_to_abs_mode[mode_id] = abs_mode_id_2
        self.abs_mode_parents[abs_mode_id_1] = []
        self.abs_mode_neighbors[abs_mode_id_1] = []
        self.abs_mode_parents[abs_mode_id_2] = []
        self.abs_mode_neighbors[abs_mode_id_2] = []

        ######### splitting parents edges ##############
        for abs_edge_id in list(self.abs_mode_parents[abs_mode_id]):
            abs_edge = self.abs_edge_list[abs_edge_id]
            #if abs_edge.source == 3 and abs_edge.dest == 1:
            #    #pdb.set_trace()
            if abs_edge.source == abs_edge.dest:
                # print("skipping abstract edge [", abs_edge.source, ", ", abs_edge.dest, "]")
                continue
            abs_guard_1 = []
            abs_guard_2 = []
            abs_guard_region_1 = pc.Region(list_poly=[])
            abs_guard_region_2 = pc.Region(list_poly=[])
            for edge_info in abs_edge.guard:
                if self.edge_list[edge_info[3]].dest in orig_mode_list_1:
                    abs_guard_1.append(edge_info)
                    #print("going to the original parent edge [", self.edge_list[edge_info[3]].source,
                    #      ", ", self.edge_list[edge_info[3]].dest, "]")
                    abs_guard_region_1 = PolyUtils.get_region_union(abs_guard_region_1, pc.box2poly(edge_info[0].T))
                elif self.edge_list[edge_info[3]].dest in orig_mode_list_2:
                    abs_guard_2.append(edge_info)
                    #print("going to the new parent edge [", self.edge_list[edge_info[3]].source,
                    #      ", ", self.edge_list[edge_info[3]].dest, "]")
                    abs_guard_region_2 = PolyUtils.get_region_union(abs_guard_region_2, pc.box2poly(edge_info[0].T))
                else:
                    pdb.set_trace()

            if len(abs_guard_1) + len(abs_guard_2) != len(abs_edge.guard):
                pdb.set_trace()

            # self.abs_edge_list[abs_edge_id].guard = abs_guard_1
            # self.abs_edge_list[abs_edge_id].region_guard = abs_guard_region_1
            if len(abs_guard_1) > 0:
                ## delete edge if it's no longer useful
                self.abs_edge_list.append(Edge(abs_edge.source, abs_mode_id_1,
                                               len(self.abs_edge_list), abs_guard_1, abs_guard_region_1))
                self.abs_mode_parents[abs_mode_id_1].extend([len(self.abs_edge_list) - 1])
                self.abs_mode_neighbors[abs_edge.source].extend([len(self.abs_edge_list) - 1])
            if len(abs_guard_2) > 0:
                self.abs_edge_list.append(Edge(abs_edge.source, abs_mode_id_2,
                                               len(self.abs_edge_list), abs_guard_2, abs_guard_region_2))
                self.abs_mode_parents[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])
                self.abs_mode_neighbors[abs_edge.source].extend([len(self.abs_edge_list) - 1])

        """
        for abs_edge_id in self.abs_mode_parents[abs_mode_id]:
            abs_edge = self.abs_edge_list[abs_edge_id]
            print("######abs_mode_id: The edges of the abstract edge: [", abs_edge.source, ", ", abs_edge.dest, "] are: ")
            for edge_info in abs_edge.guard:
                print("[", self.edge_list[edge_info[3]].source,
                      ", ", self.edge_list[edge_info[3]].dest, "]")
        for abs_edge_id in self.abs_mode_parents[abs_mode_id_2]:
            abs_edge = self.abs_edge_list[abs_edge_id]
            print("######abs_mode_id_2:  The edges of the abstract edge: [", abs_edge.source, ", ", abs_edge.dest, "] are: ")
            for edge_info in abs_edge.guard:
                print("[", self.edge_list[edge_info[3]].source,
                      ", ", self.edge_list[edge_info[3]].dest, "]")
        """


        ######### splitting children edges #############
        for abs_edge_id in list(self.abs_mode_neighbors[abs_mode_id]):
            abs_edge = self.abs_edge_list[abs_edge_id]
            if abs_edge.source == abs_edge.dest:
                # print("skipping abstract edge [", abs_edge.source, ", ", abs_edge.dest, "]")
                continue
            abs_guard_1 = []
            abs_guard_2 = []
            abs_guard_region_1 = pc.Region(list_poly=[])
            abs_guard_region_2 = pc.Region(list_poly=[])
            for edge_info in abs_edge.guard:
                if self.edge_list[edge_info[3]].source in orig_mode_list_1:
                    abs_guard_1.append(edge_info)
                    #print("going to the original child edge [", self.edge_list[edge_info[3]].source,
                    #      ", ", self.edge_list[edge_info[3]].dest, "]")
                    abs_guard_region_1 = PolyUtils.get_region_union(abs_guard_region_1, pc.box2poly(edge_info[0].T))
                elif self.edge_list[edge_info[3]].source in orig_mode_list_2:
                    abs_guard_2.append(edge_info)
                    # print("going to the new child edge [", self.edge_list[edge_info[3]].source,
                    #      ", ", self.edge_list[edge_info[3]].dest, "]")
                    abs_guard_region_2 = PolyUtils.get_region_union(abs_guard_region_2, pc.box2poly(edge_info[0].T))
                else:
                    pdb.set_trace()

            if len(abs_guard_1) + len(abs_guard_2) != len(abs_edge.guard):
                pdb.set_trace()

            # self.abs_edge_list[abs_edge_id].guard = abs_guard_1
            # self.abs_edge_list[abs_edge_id].region_guard = abs_guard_region_1
            # if len(abs_guard_1) == 0:
            #     ## delete edge if it's no longer useful
            #     if not abs_edge_id in self.abs_mode_parents[abs_edge.dest]:
            #         pdb.set_trace()
            #     self.abs_mode_parents[abs_edge.dest].remove(abs_edge_id)
            #     if not abs_edge_id in self.abs_mode_neighbors[abs_mode_id]:
            #         pdb.set_trace()
            #     self.abs_mode_neighbors[abs_mode_id].remove(abs_edge_id)
            if len(abs_guard_1) > 0:
                self.abs_edge_list.append(Edge(abs_mode_id_1, abs_edge.dest,
                                               len(self.abs_edge_list), abs_guard_1, abs_guard_region_1))
                self.abs_mode_parents[abs_edge.dest].extend([len(self.abs_edge_list) - 1])
                self.abs_mode_neighbors[abs_mode_id_1].extend([len(self.abs_edge_list) - 1])

            if len(abs_guard_2) > 0:
                self.abs_edge_list.append(Edge(abs_mode_id_2, abs_edge.dest,
                                               len(self.abs_edge_list), abs_guard_2, abs_guard_region_2))
                self.abs_mode_parents[abs_edge.dest].extend([len(self.abs_edge_list) - 1])
                self.abs_mode_neighbors[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])

        ########## handling self edges ###################
        for abs_edge_id in list(self.abs_mode_neighbors[abs_mode_id]):
            abs_edge = self.abs_edge_list[abs_edge_id]
            if abs_edge.source == abs_edge.dest:
                abs_guard_11 = []
                abs_guard_12 = []
                abs_guard_21 = []
                abs_guard_22 = []
                abs_guard_region_11 = pc.Region(list_poly=[])
                abs_guard_region_12 = pc.Region(list_poly=[])
                abs_guard_region_21 = pc.Region(list_poly=[])
                abs_guard_region_22 = pc.Region(list_poly=[])
                for edge_info in abs_edge.guard:
                    if (self.edge_list[edge_info[3]].source in orig_mode_list_1) and (self.edge_list[edge_info[3]].dest in orig_mode_list_1):
                        abs_guard_11.append(edge_info)
                        abs_guard_region_11 = PolyUtils.get_region_union(abs_guard_region_11, pc.box2poly(edge_info[0].T))
                    elif (self.edge_list[edge_info[3]].source in orig_mode_list_1) and (self.edge_list[edge_info[3]].dest in orig_mode_list_2):
                        abs_guard_12.append(edge_info)
                        abs_guard_region_12 = PolyUtils.get_region_union(abs_guard_region_12, pc.box2poly(edge_info[0].T))
                    elif (self.edge_list[edge_info[3]].source in orig_mode_list_2) and (self.edge_list[edge_info[3]].dest in orig_mode_list_1):
                        abs_guard_21.append(edge_info)
                        abs_guard_region_21 = PolyUtils.get_region_union(abs_guard_region_21, pc.box2poly(edge_info[0].T))
                    elif (self.edge_list[edge_info[3]].source in orig_mode_list_2) and (self.edge_list[edge_info[3]].dest in orig_mode_list_2):
                        abs_guard_22.append(edge_info)
                        abs_guard_region_22 = PolyUtils.get_region_union(abs_guard_region_22, pc.box2poly(edge_info[0].T))

                if len(abs_guard_11) + len(abs_guard_12) + len(abs_guard_21) + len(abs_guard_22) != len(abs_edge.guard):
                    pdb.set_trace()

                self.abs_edge_list[abs_edge_id].guard = abs_guard_11
                self.abs_edge_list[abs_edge_id].region_guard = abs_guard_region_11
                # if len(abs_guard_11) == 0:
                #     ## delete edge if it's no longer useful
                #     if not abs_edge_id in self.abs_mode_parents[abs_mode_id]:
                #         pdb.set_trace()
                #     self.abs_mode_parents[abs_mode_id].remove(abs_edge_id)
                #     if not abs_edge_id in self.abs_mode_neighbors[abs_mode_id]:
                #         pdb.set_trace()
                #     self.abs_mode_neighbors[abs_mode_id].remove(abs_edge_id)
                if len(abs_guard_11) > 0:
                    self.abs_edge_list.append(Edge(abs_mode_id_1, abs_mode_id_1,
                                                   len(self.abs_edge_list), abs_guard_11, abs_guard_region_11))
                    self.abs_mode_parents[abs_mode_id_1].extend([len(self.abs_edge_list) - 1])
                    self.abs_mode_neighbors[abs_mode_id_1].extend([len(self.abs_edge_list) - 1])

                if len(abs_guard_12) > 0:
                    self.abs_edge_list.append(Edge(abs_mode_id_1, abs_mode_id_2,
                                                   len(self.abs_edge_list), abs_guard_12, abs_guard_region_12))
                    self.abs_mode_parents[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])
                    self.abs_mode_neighbors[abs_mode_id_1].extend([len(self.abs_edge_list) - 1])
                if len(abs_guard_21) > 0:
                    self.abs_edge_list.append(Edge(abs_mode_id_2, abs_mode_id_1,
                                                   len(self.abs_edge_list), abs_guard_21, abs_guard_region_21))
                    self.abs_mode_parents[abs_mode_id_1].extend([len(self.abs_edge_list) - 1])
                    self.abs_mode_neighbors[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])
                if len(abs_guard_22) > 0:
                    self.abs_edge_list.append(Edge(abs_mode_id_2, abs_mode_id_2,
                                                   len(self.abs_edge_list), abs_guard_22, abs_guard_region_22))
                    self.abs_mode_parents[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])
                    self.abs_mode_neighbors[abs_mode_id_2].extend([len(self.abs_edge_list) - 1])

        ###### initializing caches for the new mode ##############
        # self.clean_caches()
        for abs_edge in self.abs_edge_list:
            print("The edges of the abstract edge: [", abs_edge.source, ", ", abs_edge.dest, "] are: ")
            for edge_info in abs_edge.guard:
                print("[", self.edge_list[edge_info[3]].source,
                      ", ", self.edge_list[edge_info[3]].dest, "]")
                if not self.edge_list[edge_info[3]].source in self.abs_mode_to_mode[abs_edge.source]:
                    pdb.set_trace()
                if not self.edge_list[edge_info[3]].dest in self.abs_mode_to_mode[abs_edge.dest]:
                    pdb.set_trace()

    def refine_noreplace(self, abs_mode_id: int, unsafe_inter_list: List[int]):
        if len(self.abs_mode_to_mode[abs_mode_id]) > 1:
            print("Refining mode", abs_mode_id)
            self.decompose_abs_mode_nonReplace(abs_mode_id, unsafe_inter_list)
            print("Done refinement of mode", abs_mode_id)
            return True
        else:
            print("Can't refine anymore mode", abs_mode_id)
            return False

    def refine(self, abs_mode_id: int, unsafe_inter_list: List[int]):
        if len(self.abs_mode_to_mode[abs_mode_id]) > 1:
            print("Refining mode", abs_mode_id)
            self.decompose_abs_mode(abs_mode_id, unsafe_inter_list)
            print("Done refinement of mode", abs_mode_id)
            return True
        else:
            print("Can't refine anymore mode", abs_mode_id)
            return False

    # Function that returns the trace for an agent's initial set the time bound in goal_waypoint is in global time,
    # so we need to know what is the time bound from the previous waypoint hence, time_bound =
    # goal_waypoint.time_bound - previous goal_waypoint.time_bound
    def simulate_segment(self, initial_point: np.array, goal_waypoint: Waypoint, time_step: int, time_bound: int):
        assert type(self.dynamics) == str, "must provide path"
        trace = self.TC_Simulate(goal_waypoint, time_step,
                                 initial_point)
        return trace

    def sample_initset(self):
        box = pc.bounding_box(self.initial_set)
        upper = box[1]
        lower = box[0]
        point = []

        for i in range(upper.shape[0]):
            tmp = random.uniform(lower[i],upper[i])[0]
            point.append(tmp)
        return point

class TubeMaster:
    # TODO support multiple TubeCaches
    # TODO support multiple grids for each TubeCache
    # grid_resolution is the quantization resolution

    def __init__(self, grid_resolution: np.array, dynamics_types: Dict[str, str],
                 agents_list: List[Agent], sym_level) -> None:

        self.tube_tools: Dict[str, Tuple[TubeComputer, TubeCache]] = {
            dynamics: TubeCache(TubeComputer(dynamics_types[dynamics]), sym_level) for
            dynamics in dynamics_types}
        self.grid_resolution: np.array = grid_resolution
        self.sym_level = sym_level
        # TODO support other kinds of discrepancy functions, etc.

    @staticmethod
    def quantize_uniform(center: np.array, delta: np.array) -> np.array:
        if len(center.shape) == 1 and len(delta.shape) == 1:
            raise TypeError("Quantize Uniform only accepts 1d numpy arrays")
        return np.floor(center / delta, dtype=np.int)




    def get_tube(self, cur_agent: Agent, initset: pc.Polytope, waypoint: Waypoint,
                 time_step: float) -> Tuple[List[pc.Polytope], List[np.array],
                                            List[int], List[int], List[pc.Polytope], float, int]:
        curr_tube_cache: TubeCache
        curr_tube_cache = self.tube_tools[cur_agent.dynamics]

        ########################################################
        '''
        transform_time = 0
        t = time.time()
        transform_time += time.time() - t
        '''
        ######################################################
        min_index = []
        max_index = []
        next_initset = []
        reachtube_computation_time = 0
        if not cur_agent.reached_fixed_point:
            if self.sym_level == 2:
                cur_agent.uncovered_sets_per_mode[waypoint.id] = PolyUtils.get_region_union(
                    cur_agent.uncovered_sets_per_mode[waypoint.id], initset)
                # cur_agent.uncovered_sets_per_mode[waypoint.id] = \
                #    pc.box2poly(PolyUtils.get_region_bounding_box(cur_agent.uncovered_sets_per_mode[waypoint.id]).T)
                initset = cur_agent.uncovered_sets_per_mode[waypoint.id]
            if self.sym_level == 2 and ((not pc.is_empty(cur_agent.abs_mode_initset[waypoint.id]))\
                                    and pc.is_subset(initset, cur_agent.abs_mode_initset[waypoint.id])):
                tube_list = cur_agent.abs_mode_reachset[waypoint.id]
                # trace_list = None
                trace_list = cur_agent.abs_mode_trace[waypoint.id]
            else:
                tube_list, trace_list, reach_time = \
                    curr_tube_cache.get(
                        cur_agent, initset, waypoint, time_step,
                        self.grid_resolution)
                reachtube_computation_time += reach_time
            cur_agent.number_of_tubes_computed += len(tube_list)
            if self.sym_level != 2:
                for edge_ind in cur_agent.mode_neighbors[waypoint.id]:
                    possible_next_initset_reg = pc.Region(list_poly=[])
                    curr_min_index = math.inf
                    curr_max_index = -1
                    for tube in tube_list:
                        curr_tube_min_index, curr_tube_max_index, curr_next_initset_u = TubeMaster.intersect_waypoint(
                                tube, cur_agent.edge_list[edge_ind].guard)
                        if curr_tube_max_index == -1:
                            print(tube[-1], cur_agent.edge_list[edge_ind].guard, waypoint.mode_parameters)
                            # pdb.set_trace()
                            continue
                        curr_min_index = min(curr_min_index, curr_tube_min_index)
                        curr_max_index = max(curr_max_index, curr_tube_max_index)
                        possible_next_initset_reg = PolyUtils.get_region_union(possible_next_initset_reg,
                                                                               pc.box2poly(curr_next_initset_u.T))
                    min_index.append(curr_min_index)
                    max_index.append(curr_max_index)
                    next_initset.append(possible_next_initset_reg)
            else:
                cur_agent.abs_per_virtual_mode_tube_counter[waypoint.id] += 1
                cur_agent.initset_union = PolyUtils.get_region_union(cur_agent.initset_union,
                                                                     initset)
                # cur_agent.abs_initset_rects[waypoint.id].append(initset_u)
                cur_agent.abs_mode_initset[waypoint.id] = PolyUtils.get_region_union(
                    cur_agent.abs_mode_initset[waypoint.id],
                    initset)
                # cur_agent.uncovered_rects_per_mode[waypoint.id] = []
                if len(cur_agent.abs_mode_reachset[waypoint.id]) > 0:
                    cur_agent.abs_mode_reachset[waypoint.id].extend(
                        tube_list)
                else:
                    cur_agent.abs_mode_reachset[waypoint.id] = tube_list
                cur_agent.abs_mode_trace[waypoint.id] = trace_list
                cur_agent.uncovered_sets_per_mode[waypoint.id] = pc.Region(list_poly=[])
                for edge_ind in cur_agent.abs_mode_neighbors[waypoint.id]:
                    next_mode_id = cur_agent.abs_edge_list[edge_ind].dest
                    curr_min_index, curr_max_index, possible_next_initset_reg = \
                        TubeMaster.tube_list_intersect_waypoint(cur_agent, tube_list, cur_agent.abs_edge_list[edge_ind])
                    min_index.append(curr_min_index)
                    max_index.append(curr_max_index)
                    next_initset.append([pc.Region(list_poly=[]), next_mode_id])
                    # possible_next_initset_rect = PolyUtils.get_region_bounding_box(possible_next_initset_reg)
                    # this is where we were combining all initial sets together which caused the blow up in the size of the reachtube
                    # possible_next_initset_reg = pc.box2poly(possible_next_initset_rect.T)
                    # TODO: replace the following with subtract regions from PolyUtils
                    if not pc.is_subset(possible_next_initset_reg, \
                            cur_agent.abs_mode_initset[next_mode_id]) or \
                            pc.is_empty(cur_agent.abs_mode_initset[next_mode_id]):
                        if type(possible_next_initset_reg) == pc.Polytope:
                            possible_next_initset_reg = pc.Region(list_poly=[possible_next_initset_reg])
                        for new_poly in possible_next_initset_reg.list_poly:
                            possible_next_initset_rect = PolyUtils.get_bounding_box(new_poly)
                            rect_contain = False
                            if type(cur_agent.abs_mode_initset[next_mode_id]) == pc.Polytope:
                                cur_agent.abs_mode_initset[next_mode_id] = pc.Region(list_poly=[cur_agent.abs_mode_initset[next_mode_id]])
                            for old_poly in cur_agent.abs_mode_initset[next_mode_id].list_poly:
                                existing_initset_rect = PolyUtils.get_bounding_box(old_poly)
                                if PolyUtils.does_rect_contain(possible_next_initset_rect, existing_initset_rect):
                                    rect_contain = True
                                    break
                            if not rect_contain:
                                cur_agent.uncovered_sets_per_mode[next_mode_id] = PolyUtils.get_region_union(
                                    cur_agent.uncovered_sets_per_mode[next_mode_id],
                                    new_poly)
                                next_initset[-1][0] = PolyUtils.get_region_union(
                                    next_initset[-1][0],
                                    new_poly)
                                # cur_agent.uncovered_rects_per_mode[next_mode_id].append(possible_next_initset_rect)
                        if not pc.is_subset(possible_next_initset_reg, cur_agent.initset_union):
                            cur_agent.uncovered_sets = PolyUtils.get_region_union(cur_agent.uncovered_sets, possible_next_initset_reg)

                    """
                    if not pc.is_empty(next_initset[-1]):
                        print("next_initset[-1]: ",  PolyUtils.get_region_bounding_box(next_initset[-1]))
                    else:
                        print("initial set is empty")
                    """

                    new_uncovered_sets = pc.Region(list_poly=[])
                    if type(cur_agent.uncovered_sets) == pc.Polytope:
                        cur_agent.uncovered_sets = pc.Region(list_poly=[cur_agent.uncovered_sets])
                    for i in range(len(cur_agent.uncovered_sets.list_poly)):
                        if not pc.is_subset(cur_agent.uncovered_sets.list_poly[i], cur_agent.initset_union):
                            new_uncovered_sets = PolyUtils.get_region_union(new_uncovered_sets, cur_agent.uncovered_sets.list_poly[i])
                    cur_agent.uncovered_sets = new_uncovered_sets
                # cur_agent.reached_fixed_point = cur_agent.check_fixed_point()
        else:
            tube_list = cur_agent.abs_mode_reachset[waypoint.id]
            trace_list = cur_agent.abs_mode_trace[waypoint.id]

        # t = time.time()
        if self.sym_level != 2:
            num_neighbors = len(cur_agent.mode_neighbors[waypoint.id])
        else:
            num_neighbors = len(cur_agent.abs_mode_neighbors[waypoint.id])
        if num_neighbors > 0 and not cur_agent.reached_fixed_point:
            if (len(min_index) == 0 or max(min_index) == -1): # pc.is_empty(next_initset):
                print("next initset should not be empty, system does not reach the next mode's guard of waypoint:", waypoint.mode_parameters)
                pdb.set_trace()
            else:
                if len(min_index) == 0:
                    pdb.set_trace()
                # print("min_index: ", min_index)
                # print("max_index: ", max_index)
                # print("next_initiset: ", next_initset)
                # tube = tube[:max(min_index)]
        # else:
        result_tube_list = []
        for tube in tube_list:
            result_tube_list.append([pc.box2poly(tube[i][:, :].T) for i in range(len(tube))])
        transform_time = 0
        return result_tube_list, tube_list, trace_list, min_index, max_index, next_initset, transform_time, cur_agent.reached_fixed_point, reachtube_computation_time

    @staticmethod
    def plot_segment(curr_tube: List[np.array], is_unified: bool):
        plt.figure()
        curr_tube = [tube[:, 1:] for tube in curr_tube]
        currentAxis = plt.gca()
        bigger_box = None
        if is_unified:
            for i in range(len(curr_tube)):
                box_of_poly = curr_tube[i]
                bigger_box = box_of_poly if bigger_box is None else np.row_stack(
                    (np.minimum(box_of_poly[0, :], bigger_box[0, :]),
                     np.maximum(box_of_poly[1, :], bigger_box[1, :])))
                rect = Rectangle(box_of_poly[0, [0, 1]], box_of_poly[1, 0] - box_of_poly[0, 0],
                                 box_of_poly[1, 1] - box_of_poly[0, 1], linewidth=1, facecolor='red')
                currentAxis.add_patch(rect)
            plt.xlim(bigger_box[0, 0] * 1.1, max(bigger_box[1, 0] * 1.1, 0))
            plt.ylim(bigger_box[0, 1] * 1.1, max(bigger_box[1, 1] * 1.1, 0))

        #plt.ylim([-2, 14])
        # plt.xlim([-5, 25])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    @staticmethod
    def plot_segments(curr_tubes: List[List[pc.Region]], is_unified: bool):
        plt.figure()
        currentAxis = plt.gca()
        bigger_box = None
        if is_unified:
            for curr_tube in curr_tubes:
                for i in range(len(curr_tube)):
                    box_of_poly = PolyUtils.get_region_bounding_box(curr_tube[i])
                    bigger_box = box_of_poly if bigger_box is None else np.row_stack(
                        (np.minimum(box_of_poly[0, :], bigger_box[0, :]),
                         np.maximum(box_of_poly[1, :], bigger_box[1, :])))
                    rect = Rectangle(box_of_poly[0, [0, 2]], box_of_poly[1, 0] - box_of_poly[0, 0],
                                     box_of_poly[1, 2] - box_of_poly[0, 2], linewidth=1, facecolor='red')
                    currentAxis.add_patch(rect)
            plt.xlim(bigger_box[0, 0] * 1.1, max(bigger_box[1, 0] * 1.1, 0))
            plt.ylim(bigger_box[0, 1] * 1.1, max(bigger_box[1, 1] * 1.1, 0))
        else:
            raise NotImplementedError("No support for original tubes in plot segments")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    @staticmethod
    def intersect_waypoint(tube: List[np.array], guard: np.array) -> Tuple[int, int, np.array]:
        min_index: int = -1
        max_index: int = -1
        i: int
        is_empty = True
        inter_list = []
        guard_contains_tube = False
        tube_contains_guard = False

        num_of_rects_after_inter = 15
        counter = 0
        for i in range(len(tube)):
            if PolyUtils.do_rects_inter(tube[i], guard):
                inter_list.append(PolyUtils.get_rects_inter(tube[i], guard))
                is_empty = False
                if min_index == -1:
                    min_index = i
                if i > max_index:
                    max_index = i
                if counter == num_of_rects_after_inter:
                    break
                counter += 1

        if is_empty:
            return -1,-1,np.array([[],[]])
            TubeMaster.plot_segment(tube, True)
            pdb.set_trace()

        next_initset: np.array = PolyUtils.get_convex_union(inter_list)
        if guard_contains_tube:
            next_initset = tube[i][:, :]
            max_index = i
            if min_index == -1:
                min_index = i
        elif tube_contains_guard:
            next_initset = np.copy(guard)
            max_index = i
            if min_index == -1:
                min_index = i
        return min_index, max_index, next_initset

    @staticmethod
    def tube_list_intersect_waypoint(cur_agent: Agent, tube_list: List[List[np.array]], edge: Edge):
        possible_next_initset_reg = pc.Region(list_poly=[])
        curr_min_index = math.inf
        curr_max_index = -1

        for tube in tube_list:
            curr_tube_min_index, curr_tube_max_index, curr_next_initset_u = TubeMaster.intersect_waypoint(
                tube, PolyUtils.get_region_bounding_box(edge.region_guard))
            if curr_tube_max_index == -1:
                # pdb.set_trace()
                continue
            curr_min_index = min(curr_min_index, curr_tube_min_index)
            curr_max_index = max(curr_max_index, curr_tube_max_index)
            for i in range(len(edge.guard)):
                new_next_initset_reg = cur_agent.transform_poly_to_virtual(
                    cur_agent.transform_poly_from_virtual(
                        pc.box2poly(curr_next_initset_u.T),
                        edge.guard[i][1]),
                    edge.guard[i][2])
                possible_next_initset_reg = PolyUtils.get_region_union(possible_next_initset_reg, new_next_initset_reg)
        return curr_min_index, curr_max_index, possible_next_initset_reg

    @staticmethod
    def intersect_waypoint_list(tube: List[np.array],
                                guard: List[Tuple[np.array, Tuple[float, ...], Tuple[float, ...]]]) -> np.array:
        i: int
        is_empty = True
        result_list = []
        min_index: int = -1
        max_index: int = -1
        guard_contains_tube = False
        tube_contains_guard = False
        for j in range(len(guard)):
            inter_list = []
            for i in range(len(tube)):
                if PolyUtils.do_rects_inter(tube[i], guard[j][0]):
                    inter_list.append(PolyUtils.get_rects_inter(tube[i], guard[j][0]))
                    is_empty = False
                    if min_index == -1:
                        min_index = i
                    if i > max_index:
                        max_index = i
            result_list.append((PolyUtils.get_convex_union(inter_list), guard[j][1], guard[j][2]))

        if is_empty:
            TubeMaster.plot_segment(tube, True)
            pdb.set_trace()
        else:
            pass
        return min_index, max_index, result_list

    def get_tubesegment(self, cur_agent: Agent, initset: pc.Polytope, waypoint: Waypoint,
                        time_step: float) -> ReachtubeSegment:
        # it would use the unsafeset for refinement
        tube: np.array
        trace: np.array
        min_index: int
        max_index: int
        next_initset: pc.array
        tube_list, rect_tube_list, traces_list, min_index, max_index, next_initset, transform_time, fixed_point, reachtube_computation_time = self.get_tube(cur_agent,
                                                                                                      initset, waypoint,
                                                                                                      time_step)
        # TODO optimize this with PolyUtils function to skip box2poly
        # change for TAC, uncomment later
        # tube = tube[:max_index]
        # if not fixed_point:
        #     #if traces_list is None or traces_list[-1] is None or traces_list[-1][0] is None:
        #     trace = traces_list
        #     #else:
        #     #    trace = traces_list[-1][0][:max_index, :]
        # else:
        #     trace = []
        trace = traces_list
        output: ReachtubeSegment = ReachtubeSegment(tube_list, rect_tube_list, trace, min_index,
                                                    max_index, next_initset, waypoint.id)
        cur_agent.reachtube_time += reachtube_computation_time
        return output, transform_time, fixed_point

    @staticmethod
    def trim_trace(trace: np.array, waypoint: Waypoint) -> Optional[np.array]:
        last_reached: int = -1
        i: int
        for i in range(len(trace)):
            if Agent.reached_way_point(trace[i], waypoint):
                last_reached: int = i
        new_trace: Optional[np.array] = None
        if last_reached > -1:
            new_trace: Optional[np.array] = trace[:last_reached, :]
        return new_trace


# This class would compute tubes as lists of hyper-rectangles
class TubeComputer:
    def __init__(self, reachability_engine: str):
        self.reachability_engine: str = reachability_engine
        pass

    def compute_tube(self, cur_agent: Agent, initset: np.array, waypoint: Waypoint,
                     time_step: int) -> Tuple[np.array, List[np.array]]:
        # compute trace of the agent from the center
        # bloat the simulation to tube
        # refine based on the unsafe set
        # get intersection with the waypoint
        if self.reachability_engine == "default":

            # dryvr_path = "/Users/husseinsibai/Desktop/DryVR_0.2/"  # TODO: change to be an input
            # /Users/husseinsibai/Desktop/multi-drone_simulator/
            dryvr_path = "multvr/"#  "/Users/husseinsibai/Desktop/NN_tools/multvr/"
            if initset.shape[1] == 6:
                dynamics_path = "examples/models/NNquadrotor" #NNcar" # NNquadrotor
                json_output_file = "NNquadrotor.json" # "NNcar.json" # NNquadrotor
            else:
                dynamics_path = "examples/models/NNcar"  # NNcar" # NNquadrotor
                json_output_file = "NNcar.json"  # "NNcar.json" # NNquadrotor
            # dynamics_path = "examples/Linear3D"
            # json_output_file = "Linear3D.json"
            dynamics_path = cur_agent.dynamics
            json_output_file = "NNcar_noNN.json"
            # dynamics_path = "examples/NNquadrotor_noNN"
            # json_output_file = "NNquadrotor_noNN.json"


            dryvrutils = DryVRUtils(cur_agent.variables_list, dryvr_path, dynamics_path, json_output_file, cur_agent.seed)
            dryvrutils.construct_mode_dryvr_input_file(initset, waypoint)
            tube, trace = dryvrutils.run_dryvr() # (dryvr_path, json_output_file)
            # tube = dryvrutils.parse_dryvr_output(cur_agent.variables_list, dryvr_path)
            if json_output_file == "NNcar_noNN.json" or json_output_file == "NNcar.json":
                if tube[-1][0,-1] > 6.28 or tube[-1][1,-1] > 6.28:
                    print("Round down angles")
                    for i in range(len(tube)):
                        tube[i][:,-1] = tube[i][:,-1] - np.pi*2
                elif tube[-1][0,-1] < -6.28 or tube[-1][1,-1] < -6.28:
                    print("Round up angles")
                    for i in range(len(tube)):
                        tube[i][:,-1] = tube[i][:,-1] + np.pi*2
            print(">>>", initset, tube[-1])
            return trace, tube




            k: np.array
            gamma: np.array
            k, gamma, center_trace = DiscrepancyLearning.compute_k_and_gamma(initset, waypoint, time_step,
                                                                             waypoint.time_bound, cur_agent)
            discrepancy = np.row_stack((k, gamma))  # replace it with the discrepancy computer
            # init_delta_array = (initset_box[1, :] - initset_box[0, :]) / 2
            # print("computetube_initset: ", initset)
            # print("center trace: ", center_trace[0])
            init_delta_array = (initset[1, :] - initset[0, :]) / 2
            tube = TubeComputer.bloat_to_tube(discrepancy, init_delta_array, center_trace)
            return center_trace, tube
        elif self.reachability_engine == "flowstar":
            flow_params = cur_agent.get_flowstar_parameters(waypoint.mode_parameters, initset, time_step,
                                                            waypoint.time_bound, waypoint.mode)
            print(f"start computing flowstar tube for mode {waypoint.id}")
            tube = TubeComputer.get_flowstar_tube(flow_params)

            if initset.shape[1] == 6:
                dynamics_path = "examples/NNquadrotor" #NNcar" # NNquadrotor
                json_output_file = "NNquadrotor.json" # "NNcar.json" # NNquadrotor
            else:
                dynamics_path = "examples/NNcar"  # NNcar" # NNquadrotor
                json_output_file = "NNcar.json"  # "NNcar.json" # NNquadrotor

            if json_output_file == "NNcar_noNN.json" or json_output_file == "NNcar.json":
                if tube[-1][0,-1] > 6.28 or tube[-1][1,-1] > 6.28:
                    print("Round down angles")
                    for i in range(len(tube)):
                        tube[i][:,-1] = tube[i][:,-1] - np.pi*2
                elif tube[-1][0,-1] < -6.28 or tube[-1][1,-1] < -6.28:
                    print("Round up angles")
                    for i in range(len(tube)):
                        tube[i][:,-1] = tube[i][:,-1] + np.pi*2
            print(">>>", initset, tube[-1])
            return [], tube

        elif self.reachability_engine == "sherlock":
            sherlock_params = cur_agent.get_sherlock_parameters(waypoint.mode_parameters, initset, time_step,
                                                            waypoint.time_bound, waypoint.mode)
            tube = TubeComputer.get_sherlock_tube(sherlock_params)

            return None, tube
        elif self.reachability_engine == "verisig":
            initial_condition, dynamics_string = cur_agent.get_flowstar_parameters(waypoint.mode_parameters, initset,
                                                                                   time_step, waypoint.time_bound,
                                                                                   waypoint.mode)
            tube = TubeComputer.get_verisig_tube(initial_condition, dynamics_string, waypoint.time_bound)
            return None, tube

        raise NotImplementedError("System does not support other reachability types")

    # duration here is the maximum time at which the tube intersects the guard minus the minimum time it reached it.

    @staticmethod
    def bloat_to_tube(discrepancy, init_delta_array, trace):
        '''
            guard: is a polytope that represents the target of the tube
        '''
        if trace.shape[0] < 1:
            pdb.set_trace()
            raise ValueError("Trace is too small")
        k = discrepancy[0, :]
        gamma = discrepancy[1, :]
        time_intervals = trace[1:, 0] - trace[0, 0]
        deltas = np.exp(np.outer(time_intervals, gamma)) * np.tile(init_delta_array * k, (time_intervals.shape[0], 1))
        #print("init_delta_array: ", init_delta_array)
        reach_tube = np.stack((np.minimum(trace[1:, :], trace[:-1, :]) - deltas,
                               np.maximum(trace[1:, :], trace[:-1, :]) + deltas), axis=1)
        reach_tube = list(reach_tube)
        # print("computed tube initial set: ", reach_tube[0])
        return reach_tube

    @staticmethod
    def get_verisig_tube(initial_condition, dynamics_string,time_bound):
        spaceex_xml_fn = './verisig/examples/quadrotor_one_mode/quadrotor_MPC.xml'
        spaceex_cfg_fn = './verisig/examples/quadrotor_one_mode/quadrotor_MPC.cfg'
        verisig_cfg_fn = './verisig/examples/quadrotor_one_mode/quadrotor_MPC.yml'
        verisig_net_fn = './verisig/examples/quadrotor_one_mode/tanh20x20.yml'

        # dynamics_text = "x1' == x4 - 0.25 &\nx2' == x5 + 0.25 &\nx3' == x6 &\nx4' == 9.81 * sin(u1) / cos(u1) &\nx5' == - 9.81 * sin(u2) / cos(u2) &\nx6' == u3 - 9.81 &\nx7' = x4 &\nx8' = x5 &\nx9' = x6 &\nclock' == 1 &\nt' == 1"

        f = open(verisig_cfg_fn,'r')
        verisig_cfg = yaml.safe_load(f)
        i = 0
        for cond in initial_condition:
            lower = initial_condition[cond][0]
            upper = initial_condition[cond][1]
            verisig_cfg['init']['states'][i] = f"{cond} in [{lower},{upper}]"
            i+=1
        verisig_cfg['time'] = time_bound
        f.close()
        f = open(verisig_cfg_fn,'w')
        yaml.safe_dump(verisig_cfg,f)

        ET.register_namespace('', "http://www-verimag.imag.fr/xml-namespaces/sspaceex")
        tree = ET.parse(spaceex_xml_fn)
        root = tree.getroot()
        model = root[0]
        for child in model:
            if child.tag == '{http://www-verimag.imag.fr/xml-namespaces/sspaceex}location' and child.attrib['id'] == '3':
                dynamics = child[1]
                dynamics.text = dynamics_string
        tree.write(spaceex_xml_fn)
        subprocess.run(['./verisig/verisig','--flowstar-cmd',\
                        './verisig/flowstar/flowstar','-o',\
                        '-sc',spaceex_cfg_fn,'-vc',verisig_cfg_fn,\
                        spaceex_xml_fn,verisig_net_fn])

        cmd = './verisig/flowstar_plot/flowstar < ./outputs/autosig.flow'
        os.system(cmd)

        tube_fn = './outputs/autosig.plt'
        f = open(tube_fn,'r')

        trace = []
        idx_list = [27,28,29,24,25,26]
        line = f.readline()
        while line[0] != 'e':
            if not re.search('[a-zA-z]', line) and line != '':
                line = line.replace('\n','')
                line = line.split(' ')
                tmp2 = []
                tmp = []
                for idx in idx_list:
                    tmp.append(float(line[idx]))
                tmp2.append(tmp)
                line = f.readline()
                line = line.replace('\n', '')
                line = line.split(' ')
                tmp = []
                for idx in idx_list:
                    tmp.append(float(line[idx]))
                tmp2.append(tmp)
                trace.append(np.array(tmp2))

            line = f.readline()

        return trace

    @staticmethod
    # def get_flowstar_tube(params):
    #     ABS_TOL = 1e-4
    #     result = str(subprocess.check_output(["./flowstar_wrapper/Brusselator"] + params)).split('\\n')
    #     time: float = float(result[0][result[0].index(": ") + 1:])
    #     print(time)
    #     iotube = [val.replace("] [", "],! [").split(",! ") for val in result[1:-1]]
    #     my_boxes = []
    #     for poly in iotube:
    #         cur_list = []
    #         for arr in poly:
    #             cur_list.append(ast.literal_eval(arr))
    #         my_boxes.append(np.column_stack(cur_list))
    #     tube = np.stack(my_boxes)
    #     # Added to bloat convering flowtubes to be above our abs_tolerance for non-empty intersection
    #     tube[:, 0, :] -= ABS_TOL
    #     tube[:, 1, :] += ABS_TOL
    #     return tube[:, :, 1:]

    def get_flowstar_tube(params):
        model_string = params[0]
        num_variables = params[1]
        with open("flowstar_model.model",'w+') as output_file:
            output_file.write(model_string)

        cmd = './flowstar_plot/flowstar < ./flowstar_model.model'
        os.system(cmd)

        tube_fn = './outputs/flowstar_tube.plt'
        f = open(tube_fn,'r')

        trace = []
        # idx_list = [27,28,29,24,25,26]
        line = f.readline()
        while line[0] != 'e':
            if not re.search('[a-zA-z]', line) and line != '':
                line = line.replace('\n','')
                line = line.split(' ')
                line = line[1:]
                tmp2 = []
                tmp = []
                for idx in range(num_variables):
                    tmp.append(float(line[idx]))
                tmp2.append(tmp)
                line = f.readline()
                line = line.replace('\n', '')
                line = line.split(' ')
                line = line[1:]
                tmp = []
                for idx in range(num_variables):
                    tmp.append(float(line[idx]))
                tmp2.append(tmp)
                trace.append(np.array(tmp2))

            line = f.readline()

        return np.array(trace)

    @staticmethod
    def get_sherlock_tube(params):
        ABS_TOL = 1e-4
        _ = str(subprocess.check_output(["cd ../sherlockV/neural_network_reachability/examples/tests/"]
                                             ))
        result = str(subprocess.check_output(["./Brusselator"]
                                             + params)).split('\\n')
        _ = str(subprocess.check_output(["cd ../../../../src/Ye"]
                                        ))
        print("result from sherlock: ", result)
        time: float = float(result[0][result[0].index(": ") + 1:])
        print(time)
        iotube = [val.replace("] [", "],! [").split(",! ") for val in result[1:-1]]
        my_boxes = []
        for poly in iotube:
            cur_list = []
            for arr in poly:
                cur_list.append(ast.literal_eval(arr))
            my_boxes.append(np.column_stack(cur_list))
        tube = np.stack(my_boxes)
        # Added to bloat convering flowtubes to be above our abs_tolerance for non-empty intersection
        tube[:, 0, :] -= ABS_TOL
        tube[:, 1, :] += ABS_TOL
        return tube[:, :, 1:]

    # TODO implement get reachtube functions
    # TODO find better Data-structure for queries


class UniformTubeset:

    def __init__(self, input_tube=[], input_trace=None):
        self.tube: List[np.array] = input_tube
        self.trace: Optional[np.array] = input_trace


class TubeCache:

    def __init__(self, curr_tube_computer: TubeComputer, sym_level):
        self.tube_computer = curr_tube_computer
        self.tube_dict: Dict[Tuple[int, ...], UniformTubeset] = dict()
        self.computed_counter = 0
        self.saved_counter = 0
        self.sym_level = sym_level

    def get(self, cur_agent: Agent, initset_region: pc.Region, waypoint: Waypoint,
            time_step: float, grid_resolution) -> Tuple[List[np.array], List[np.array]]:

        reach_time = 0
        time_bound = waypoint.time_bound

        def next_quantized_key(curr_key: np.array, quantized_key_range: np.array) -> np.array:
            if len(curr_key.shape) > 1:
                raise ValueError("key must be one dimensional lower left and corner of bounding box")
            next_key = np.copy(curr_key)
            for dim in range(curr_key.shape[0] - 1, -1, -1):
                if curr_key[dim] < quantized_key_range[1, dim]:
                    next_key[dim] += 1
                    for reset_dim in range(dim + 1, curr_key.shape[0]):
                        next_key[reset_dim] = quantized_key_range[0, reset_dim]
                    return next_key
            raise ValueError("curr_key should not exceed the bounds of the bounding box.")

        if type(initset_region) == pc.Polytope:
            initset_region = pc.Region(list_poly=[initset_region])

        final_tube: List[List[np.array]] = []
        accumulated_traces: List[List[np.array]] = []
        if not initset_region.list_poly:
            print("*** Things not right ***")
            new_initset_region = pc.Region(list_poly=[])
        else:
            new_initset_region = pc.Region(list_poly=[initset_region.list_poly[0]])
        visited_list = []
        for ind, poly in enumerate(initset_region.list_poly):
            for new_ind, new_poly in enumerate(new_initset_region.list_poly):
                if not pc.is_empty(pc.intersect(poly, new_poly)):
                    new_initset_region.list_poly[new_ind] = pc.box2poly(PolyUtils.get_bounding_box(pc.union(poly,new_poly)).T)
                elif ind not in visited_list:
                    visited_list.append(ind)
                    new_initset_region.list_poly.append(poly)
        initset_region = new_initset_region

        for initset in initset_region.list_poly:
            initset_u: np.array
            if self.sym_level == 1:
                transform_information = cur_agent.get_transform_information(waypoint)
                initset_virtual = cur_agent.transform_poly_to_virtual(initset, transform_information)
                initset_u = PolyUtils.get_bounding_box(initset_virtual)
                transform_information_vir = cur_agent.get_transform_information(waypoint)
                transformed_waypoint = cur_agent.transform_mode_to_virtual(waypoint, transform_information_vir)
            else:
                initset_u = PolyUtils.get_bounding_box(initset)
                transformed_waypoint = copy.copy(waypoint)


            initset_width = initset_u[1, :] - initset_u[0, :]
            grid_delta: np.array = grid_resolution
            accumulated_tubes: List[np.array] = []
            # accumulated_traces.append([])
            counter = 0
            quantized_key_range: np.array = np.floor(initset_u / grid_delta)
            zero_indices = np.where(initset_width == 0)[0]
            num_zero_indices = zero_indices.shape[0]
            curr_key: np.array = quantized_key_range[0, :]
            while True:
                if self.sym_level != 1:
                    curr_initset: np.array = initset_u
                else:
                    curr_initset: np.array = np.row_stack((curr_key * grid_delta, curr_key * grid_delta + grid_delta))
                if num_zero_indices > 0:
                    curr_initset[0, zero_indices] = np.array(initset_u[0, zero_indices])
                    curr_initset[1, zero_indices] = np.array(initset_u[0, zero_indices])
                if PolyUtils.do_rects_inter(curr_initset, initset_u):
                    hash_key = tuple(curr_key)
                    if hash_key not in self.tube_dict or self.sym_level != 1: # TACAS 2021: change to self.sym_level == 0
                        t = time.time()
                        trace, tube = self.tube_computer.compute_tube(cur_agent, curr_initset, transformed_waypoint,
                                                                      time_step)
                        reach_time += (time.time()-t)
                        # trace = None
                        self.tube_dict[hash_key] = UniformTubeset(tube, trace)
                        self.computed_counter += 1
                    elif len(self.tube_dict[hash_key].tube) < math.ceil(time_bound / time_step):
                        transformed_waypoint.time_bound = (math.ceil(time_bound / time_step) - len(
                            self.tube_dict[hash_key].tube)) * time_step
                        t = time.time()
                        trace, tube = self.tube_computer.compute_tube(cur_agent, self.tube_dict[hash_key].tube[-1],
                                                                      transformed_waypoint, time_step)
                        reach_time += (time.time()-t)

                        # TODO extend trace as well
                        saved_fraction = len(self.tube_dict[hash_key].tube) / math.ceil(time_bound / time_step)
                        self.saved_counter += saved_fraction
                        self.computed_counter += 1 - saved_fraction
                        self.tube_dict[hash_key].tube = np.concatenate((self.tube_dict[hash_key].tube, tube), axis=0)
                    else:
                        self.saved_counter += 1
                    used_tube = self.tube_dict[hash_key].tube
                    accumulated_tubes.append(used_tube)
                    # accumulated_traces += self.tube_dict[hash_key].trace
                    counter += 1
                if np.all(curr_key == quantized_key_range[1, :]):
                    break
                if self.sym_level != 1:
                    break
                else:
                    curr_key = next_quantized_key(curr_key, quantized_key_range)

            final_tube.append(PolyUtils.merge_tubes(accumulated_tubes))

            if len(final_tube[-1]) == 0 or final_tube[-1][0][0, :].shape[0] == 0:
                pdb.set_trace()

            if self.sym_level == 1:
                final_tube[-1] = [PolyUtils.get_bounding_box(cur_agent.transform_poly_from_virtual(pc.box2poly(final_tube[-1][i][:, :].T), transform_information)) for
                             i in
                             range(len(final_tube[-1]))]

        return final_tube, accumulated_traces, reach_time
