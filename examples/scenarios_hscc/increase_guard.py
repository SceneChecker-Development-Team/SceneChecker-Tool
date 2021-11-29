import json
import numpy as np

def increase_guard(fn, new_tb = 5, old_tb = 3):
    config = {}
    with open(fn,'r') as f:
        config = json.load(f)

    guards = config["agents"][0]["guards"]
    # mode_list = config["agents"][0]["mode_list"]
    # mode_len = 
    # guard_list = config["agents"][0]["guards"]

    for i in range(len(guards)):
        # edge = edge_list[i]
        # if edge[0] == 40 and edge[1] == 45:
        #     print("here")
        # src = config["agents"][0]["mode_list"][edge[0]][1]

        # orientation = np.arctan2(src[3]-src[1],src[2]-src[0])
        # config["agents"][0]["guards"][i][1][0][2] = orientation - rad
        # config["agents"][0]["guards"][i][1][1][2] = orientation + rad
        # timeHorizons[i] = timeHorizons[i] / old_tb * new_tb
        guard = guards[i][1]
        guard[0][0] += 0.25
        guard[0][1] += 0.25
        guard[1][0] -= 0.25
        guard[1][1] -= 0.25

    config["agents"][0]["guards"] = guards

    with open(fn,'w') as f:     
        json.dump(config,f)    

if __name__ == "__main__":
    fn = "./comp2D-1_carNoNN_SymTR_50_larger.json"
    increase_guard(fn)