import json
import numpy as np

def change_time_bound(fn, new_tb = 5, old_tb = 3):
    config = {}
    with open(fn,'r') as f:
        config = json.load(f)

    timeHorizons = config["agents"][0]["timeHorizons"]
    # mode_list = config["agents"][0]["mode_list"]
    # mode_len = 
    # guard_list = config["agents"][0]["guards"]

    for i in range(len(timeHorizons)):
        # edge = edge_list[i]
        # if edge[0] == 40 and edge[1] == 45:
        #     print("here")
        # src = config["agents"][0]["mode_list"][edge[0]][1]

        # orientation = np.arctan2(src[3]-src[1],src[2]-src[0])
        # config["agents"][0]["guards"][i][1][0][2] = orientation - rad
        # config["agents"][0]["guards"][i][1][1][2] = orientation + rad
        timeHorizons[i] = timeHorizons[i] / old_tb * new_tb

    config["agents"][0]["timeHorizons"] = timeHorizons

    with open(fn,'w') as f:     
        json.dump(config,f)    

if __name__ == "__main__":
    fn = "./comp2D-1_quadNoNN_SymTR_10_flowstar.json"
    change_time_bound(fn)