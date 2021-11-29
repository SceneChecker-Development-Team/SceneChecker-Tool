import numpy as np
from src.Waypoint import Waypoint
import json
import os
import subprocess
from src.dryvr_parser import dryvr_parse
# from src.PolyUtils import PolyUtils
from typing import List
import sys
from multvr.main import get_tube


class DryVRUtils:

    def __init__(self, variables_list: List[str], dryvr_path: str, dynamics_path:str, json_output_file: str, seed = 0):
        self.variables_list = variables_list
        self.dryvr_path = dryvr_path
        self.json_output_file = json_output_file
        self.dynamics_path = dynamics_path
        self.seed = seed
        #sys.path.insert(1, self.dryvr_path)
        #from main import main

    def  construct_mode_dryvr_input_file(self, initial_set: np.array, waypoint: Waypoint):
        dryvr_vertex_list = []
        dryvr_edge_list = []
        dryvr_guards_list = []
        dryvr_resets_list = []
        dryvr_unsafeset_string = ""
        dryvr_directory_string = self.dynamics_path # "examples/Linear3D"
        dryvr_mode = str(waypoint.mode_parameters)
        dryvr_mode = dryvr_mode.replace(",", ";")
        dryvr_vertex_list.append(dryvr_mode)
        dryvr_variables_list = self.variables_list # change when we have more than one agent

        dryvr_initset_list = initial_set.tolist()
        dryvr_timehorizon = waypoint.time_bound

        # self.dryvr_path + "input/nondaginput/" +
        try:
            json_output = open(self.dryvr_path + "input/nondaginput/" + self.json_output_file, "w")
        except IOError:
            print('File does not exist')

        '''
        "edge": dryvr_edge_list,
        "guards": dryvr_guards_list,
        "resets": dryvr_resets_list,
        "unsafeSet": dryvr_unsafeset_string,
        "initialVertex": 0,
        "seed": 664891646,
        "bloatingMethod": "PW",
        "kvalue": [1] * len(dryvr_variables_list),
        '''

        json.dump({"vertex": dryvr_vertex_list,
                   "variables": dryvr_variables_list,
                   "initialSet": dryvr_initset_list,
                   "timeHorizon": dryvr_timehorizon,
                   "directory": dryvr_directory_string,
                   "seed": self.seed
                   }, json_output, indent=2)
        json_output.close()


    def run_dryvr(self):
        cur_folder_path = os.getcwd()
        # os.chdir(self.dryvr_path)
        # params = ["python3", "main.py", "input/nondaginput/" + self.json_output_file]

        # result = str(subprocess.check_output(params)).split('\\n')
        # result = main("input/nondaginput/" + self.json_output_file)
        # os.chdir(cur_folder_path)
        # from main import get_tube
        result, traces = get_tube("multvr/input/nondaginput/" + self.json_output_file)
        # os.chdir(cur_folder_path)
        # print("dryvr output: ", result)
        return result, traces


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
            if len(lb) != len(self.variables_list) or len(ub) != len(self.variables_list):
                continue
            tube.append(np.array([lb, ub]))
            # tube.append(ub)
        return tube

    def parse_dryvr_output(self):
        try:
            file = open(self.dryvr_path + "multvr/output/reachtube.txt", 'r')
        except IOError:
            print('File does not exist')
        lines = file.readlines()
        initNode, y_min, y_max = dryvr_parse(lines)

        # ydim = eval(args.y)
        # xdim = eval(args.x)
        # Using DFS algorithm to Draw image per Node
        tube = DryVRUtils.get_dryvr_tube(self.variables_list, initNode)
        return tube
        # initNode.printTube()




# if __name__ == '__main__':



