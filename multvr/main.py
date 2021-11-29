import json
import sys

from multvr.src.backend.reachabilityengine import ReachabilityEngine
from multvr.src.frontend.scenario import Scenario
from multvr.src.frontend.plotter import plot_rtsegment_and_traces
import numpy as np

def get_tube(scenario_file_name):
    cur_scenario: Scenario = Scenario(scenario_file_name)
    reachtube_segment, simulation_traces= ReachabilityEngine.get_reachtube_segment(cur_scenario.initial_set)
    # plot_rtsegment_and_traces(reachtube_segment, cur_scenario.initial_set.training_traces)
    #print("dryvr initial set has a center of ", cur_scenario.initial_set.initial_center, " and radii",
    #      cur_scenario.initial_set.initial_radii)
    return [reachtube_segment[i][:, 1:] for i in range(np.shape(reachtube_segment)[0])], [simulation_traces[i,:,1:] for i in range(np.shape(simulation_traces)[0])]

# assert ".json" in sys.argv[-1], "Please provide json input file"
# print(get_tube(sys.argv[-1]))


if __name__=='__main__':
    # take json file name from parameter
    print(get_tube("input/nondaginput/NNcar.json")) #  "input/quadrotor_test.json"

