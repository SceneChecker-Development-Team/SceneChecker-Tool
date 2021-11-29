from src.VerificationScenario import VerificationScenario
import os 
import sys
import numpy as np
import datetime
import traceback

def main():
    all_stats = []
    now = str(datetime.datetime.now())
    now = now.replace(' ','_')
    now = now.replace(':','-')
    now = now.split('.')[0]
    log_fn = f"./log/log_{now}"
    log_file_name = log_fn
    log_file = open(log_fn,'w+')
    old_stdout = sys.stdout
    for i in range(10):
        # print(f">>>>> Run {i}")
        try:
            scene = VerificationScenario(str(os.path.abspath(sys.argv[1])), log_file=log_file, seed = False)
            _,stats = scene.verify(use_dryvr=False)
            all_stats.append(stats)
        except Exception as e:
            sys.stdout = old_stdout
            # traceback.print_exc()
            # print(e)
            
    sys.stdout = old_stdout
    log_file.close()
    all_stats = np.array(all_stats)
    all_stats = np.mean(all_stats, axis = 0)
    print("###################")
    print("###################")
    print("###################")
    print("Nref", all_stats[0])
    print("|S|: ", all_stats[1])
    print("|S_v|^i: ", all_stats[2])
    print("|E_v|^i: ", all_stats[3])
    print("|S_v|^f: ", all_stats[4])
    print("|E_v|^f: ", all_stats[5])
    print("Rc: ", all_stats[6])
    print("Rt: ", all_stats[7])
    print("Tt: ", all_stats[8])

if __name__ == "__main__":
    main()