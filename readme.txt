This is the repeatability evaluation package for the tool paper "SceneChecker: Boosting Scenario Verification using Symmetry Abstractions" by: Hussein Sibai, Yangge Li, and Sayan Mitra, submitted to CAV'21.

The link to SceneChecker's website is: https://publish.illinois.edu/scenechecker/

The link to this artifact VM on Google drive is: https://drive.google.com/file/d/1FBGRxI1v4GGf-ly-B3Tv2WIRIzBo90FA/view

The link to the source code git repository for SceneChecker is: https://gitlab.engr.illinois.edu/sibai2/multi-drone_simulator/-/tree/CAV_artifact

The link to the other tools that we compare against in the paper is:
https://gitlab.engr.illinois.edu/sibai2/multi-drone_simulator/-/tree/CAV_artifact_othertools

All tools including SceneChecker with their source codes are provided with this VM. There is no tool or library that should be downloaded before usage, and all tools can be used in this VM without an internet connection.

###########################

This file explains how to replicate the results presented in the paper.

There is a typo in the last row of Table 1, for Q-S5, |S| is 188 instead of 280, while the remaining stats in the row remains unchanged. This does not affect the analysis provided in the paper.

The results of SceneChecker+Flow*, CacheReach+Flow*, and Flow* should be the same as those of the paper since their algorithms and implementations are deterministic.
 
The results of the experiments with tools that involve DryVR (i.e. SceneChecker+DryVR, CacheReach+DryVR, and DryVR) are stochastic and change between runs. The reason is that each time DryVR is called, it randomly samples traces of the system from which it computes the requested reachset.
We offer the reviewers with two options to replicate our results with these tools:

1- a fixed random seed approach: we fix the random seed for DryVR so that repeated experiments lead to the same results. The random seed we fix for the repeatability package results in the same or better (fewer number of refinements, fewer modes and edges of the abstract automaton, fewer reachsets computed, and shorter computation time) statistics by SceneChecker+DryVR than those reported in the paper. SceneChecker+DryVR still outperforms CacheReach+DryVR and DryVR satisfying the analyses in the paper. The results that are expected when using this random seed are shown at the end of this file.

2- an averaging approach: instead of fixing the random seed, the commands we provide for this approach would run the experiment of each scenario 10 times and then take the average of their statistics. This approach also leads to values that are close to those reported in the paper and satisfy the same analyses presented there. The downside for this approach is that it takes longer computation time.  Sample results from this approach are shown at the end of this file.

In general, the computation time results (Reach time (Rt) and total time (Tt)) will differ from those of the paper as they are machine specific. The machine specifications used for generating the results in the paper are shown at the end of this file.


###########################

Dependencies: All dependencies for the repeatability package are already installed in the Virtual Machine
Python 3.7 or newer versions
numpy
scipy
matplotlib
polytope
dill
ray
PyTorch

To replicate our experiments from the paper please run commands on our example files from the root directory of our local repo. Note that currently SceneChecker will visualize the computed reachsets and the visualization can be time consuming. You can kill the process after the final results appear in the terminal, if the figures are not needed.
1. For the experiment results in Table 1:
    SceneChecker+DryVR: change directory by running the following command: 
        cd ~/SceneChecker 
    and add the folder to python path by running the following command:
        export PYTHONPATH=$(pwd):$PYTHONPATH
    and then run the following commands
        Fixed random seed approach:
            C-S1: 	python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s1_dr.json
            C-S2:	python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s2_dr.json
            C-S3:	python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s3_dr.json
            C-S4.a: python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s4a_dr.json
            C-S4.b: python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s4b_dr.json
            Q-S1:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s1_dr.json
            Q-S2:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s2_dr.json
            Q-S3:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s3_dr.json
            Q-S4.a:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s4_dr.json
            Q-S5:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s5_dr.json
        Averaging Approach:
            C-S1: 	python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s1_dr.json
            C-S2:	python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s2_dr.json
            C-S3:	python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s3_dr.json
            C-S4.a: python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s4a_dr.json
            C-S4.b: python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s4b_dr.json
            Q-S1:	python3.7 src/VerificationMultiple.py examples/scenarios_cav/q_s1_dr.json
            Q-S2:	python3.7 src/VerificationMultiple.py examples/scenarios_cav/q_s2_dr.json
            Q-S3:	python3.7 src/VerificationMultiple.py examples/scenarios_cav/q_s3_dr.json
            Q-S4.a:	python3.7 src/VerificationMultiple.py examples/scenarios_cav/q_s4_dr.json
            Q-S5:	python3.7 src/VerificationMultiple.py examples/scenarios_cav/q_s5_dr.json    

    SceneChecker+Flow*: change directory to the folder by running the following command:
        cd ~/SceneChecker 
    and add the folder to python path by running the following command:
        export PYTHONPATH=$(pwd):$PYTHONPATH
    and then run the following commands
        C-S1:	python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s1_fs.json
        C-S2:	python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s2_fs.json
        C-S3:	python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s3_fs.json
        C-S4.a:	python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s4a_fs.json
        C-S4.b:	python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s4b_fs.json
        Q-S1:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s1_fs.json
        Q-S2:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s2_fs.json
        Q-S3:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s3_fs.json
        Q-S4.a:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s4_fs.json
        Q-S5:	python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s5_fs.json

    CacheReach+DryVR: change directory by running the following command: 
        cd ~/other_tools/CacheReach 
    and add the folder to python path by running the following command:
        export PYTHONPATH=$(pwd):$PYTHONPATH
    and then run the following commands
        Fixed random seed approach:
            C-S1:   python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s1_dr.json
            C-S2:   python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s2_dr.json
            C-S3:   python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s3_dr.json
            C-S4a:  python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s4a_dr.json
            C-S4b:  python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s4b_dr.json
            Q-S1:   python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s1_dr.json
            Q-S2:   python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s2_dr.json
	 Note that Q-S1 and Q-S2 timeout (more than 2 hours on the reference machine), hence we did not try the more complicated scenarios of Q-S3, Q-S4.a, and Q-S5.
	 Recall that Q-S1 is a scenario with just six segments, while Q-S3, Q-S4.a, and Q-S5 have 458, 520, and 188 segments, respectively.
	 We will update our description of NA to include timing out.
        Averaging approach:
            C-S1:   python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s1_dr.json
            C-S2:   python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s2_dr.json
            C-S3:   python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s3_dr.json
            C-S4a:  python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s4a_dr.json
            C-S4b:  python3.7 src/VerificationMultiple.py examples/scenarios_cav/c_s4b_dr.json
            Q-S1:   python3.7 src/VerificationMultiple.py examples/scenarios_cav/q_s1_dr.json
            Q-S2:   python3.7 src/VerificationMultiple.py examples/scenarios_cav/q_s2_dr.json
        
    DryVR: Change directory by running the following command: 
        cd ~/other_tools/DryVR
    and run the following commands in folder
        Fixed random seed approach:
            C-S1:	python3.7 main.py input/scenario_cav/c_s1.json
            C-S2:	python3.7 main.py input/scenario_cav/c_s2.json
            C-S3:	python3.7 main.py input/scenario_cav/c_s3.json
            C-S4.a:	python3.7 main.py input/scenario_cav/c_s4a.json
            C-S4.b:	python3.7 main.py input/scenario_cav/c_s4b.json
            Q-S1:	python3.7 main.py input/scenario_cav/q_s1.json
            Q-S2:	python3.7 main.py input/scenario_cav/q_s2.json
            Q-S3:	python3.7 main.py input/scenario_cav/q_s3.json
            Q-S4.a:	python3.7 main.py input/scenario_cav/q_s4.json
            Q-S5:	python3.7 main.py input/scenario_cav/q_s5.json
        Averaging approach:
            C-S1:	python3.7 main_multiple.py input/scenario_cav/c_s1.json
            C-S2:	python3.7 main_multiple.py input/scenario_cav/c_s2.json
            C-S3:	python3.7 main_multiple.py input/scenario_cav/c_s3.json
            C-S4.a:	python3.7 main_multiple.py input/scenario_cav/c_s4a.json
            C-S4.b:	python3.7 main_multiple.py input/scenario_cav/c_s4b.json
            Q-S1:	python3.7 main_multiple.py input/scenario_cav/q_s1.json
            Q-S2:	python3.7 main_multiple.py input/scenario_cav/q_s2.json
            Q-S3:	python3.7 main_multiple.py input/scenario_cav/q_s3.json
            Q-S4.a:	python3.7 main_multiple.py input/scenario_cav/q_s4.json
            Q-S5:	python3.7 main_multiple.py input/scenario_cav/q_s5.json


    CacheReach+Flow*: change directory by running the following command: 
        cd ~/other_tools/CacheReach 
    and add the folder to the Python path by running the following command:
        export PYTHONPATH=$(pwd):$PYTHONPATH
    and then run the following commands
        C-S1:   python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s1_fs.json
        C-S2:   python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s2_fs.json
        C-S3:   python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s3_fs.json
        C-S4.a:  python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s4a_fs.json
        C-S4.b:  python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s4b_fs.json
        Q-S1:   python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s1_fs.json
        Q-S2:   python3.7 src/VerificationScenario.py examples/scenarios_cav/q_s2_fs.json
	Note that Q-S1 and Q-S2 timeout (more than 2 hours on the reference machine), hence we did not try the more complicated scenarios of Q-S3, Q-S4.a, and Q-S5.
	Recall that Q-S1 is a scenario with just six segments, while Q-S3, Q-S4.a, and Q-S5 have 458, 520, and 188 segments, respectively.
	We will update our description of NA to include timing out.

    Flow*: change directory by running the following command:
        cd ~/other_tools/Flowstar
    and run the following commands in the folder 
        C-S1:	./flowstar < example_cav/c_s1.model
        C-S2:	./flowstar < example_cav/c_s2.model
        C-S3:	./flowstar < example_cav/c_s3.model
        C-S4.a:	./flowstar < example_cav/c_s4a.model
        C-S4.b:	./flowstar < example_cav/c_s4b.model
        Q-S1:	./flowstar < example_cav/q_s1.model
        Q-S2:	./flowstar < example_cav/q_s2.model
        Q-S3:	./flowstar < example_cav/q_s3.model
        Q-S4.a:	./flowstar < example_cav/q_s4.model
        Q-S5:	./flowstar < example_cav/q_s5.model

2. For the experiment results before Table 2:
    SceneChecker+DryVR: change directory by running the following command:
        cd ~/SceneChecker 
    and add the folder to python path by running the following command:
        export PYTHONPATH=$(pwd):$PYTHONPATH
    and then run the following commands
        QNN-S2:     python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s2.json
        QNN-S3:     python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s3.json
        QNN-S4:     python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s4.json
            
    DryVR: change directory by running the following command:
        cd ~/other_tools/DryVR
    and run the following commands in folder
        QNN-S2:     python3.7 main.py input/scenario_cav/qnn_s2.json
        QNN-S3:     python3.7 main.py input/scenario_cav/qnn_s3.json
        QNN-S4:     python3.7 main.py input/scenario_cav/qnn_s4.json
        
3. For the experiment in Table 2, change directory by running the following command:
        cd ~/SceneChecker 
and add the folder to the Python path by running the following command:
        export PYTHONPATH=$(pwd):$PYTHONPATH
and then run the following commands: 
    Fixed random seed approach:
        CNN-S2:     python3.7 src/VerificationScenario.py examples/scenarios_cav/cnn_s2TR_dr.json
        CNN-S4:     python3.7 src/VerificationScenario.py examples/scenarios_cav/cnn_s4TR_dr.json
        QNN-S2_TR: 	python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s2TR_dr.json
        QNN-S3_TR: 	python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s3TR_dr.json
        QNN-S4_TR: 	python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s4TR_dr.json
        QNN-S2_T:   python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s2T_dr.json
        QNN-S3_T:   python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s3T_dr.json
        QNN-S4_T:   python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s4T_dr.json
    Averaging approach:
        CNN-S2:     python3.7 src/VerificationMultiple.py examples/scenarios_cav/cnn_s2TR_dr.json
        CNN-S4:     python3.7 src/VerificationMultiple.py examples/scenarios_cav/cnn_s4TR_dr.json
        QNN-S2_TR: 	python3.7 src/VerificationMultiple.py examples/scenarios_cav/qnn_s2TR_dr.json
        QNN-S3_TR: 	python3.7 src/VerificationMultiple.py examples/scenarios_cav/qnn_s3TR_dr.json
        QNN-S4_TR: 	python3.7 src/VerificationMultiple.py examples/scenarios_cav/qnn_s4TR_dr.json
        QNN-S2_T:   python3.7 src/VerificationMultiple.py examples/scenarios_cav/qnn_s2T_dr.json
        QNN-S3_T:   python3.7 src/VerificationMultiple.py examples/scenarios_cav/qnn_s3T_dr.json
        QNN-S4_T:   python3.7 src/VerificationMultiple.py examples/scenarios_cav/qnn_s4T_dr.json    

4. To replicate the figures in Figure 4, change directory by running the following command:
    cd ~/SceneChecker 
and add the folder to the Python path by running the following command:
    export PYTHONPATH=$(pwd):$PYTHONPATH
and then run the following commands:
    python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s2_dr.json
    python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s3_dr.json
    python3.7 src/VerificationScenario.py examples/scenarios_cav/c_s4_dr.json

5. To replicate the figures in Figure 5, change directory by running the following command: 
    cd ~/SceneChecker 
and add the folder to Python path by running the following command:
    export PYTHONPATH=$(pwd):$PYTHONPATH
and then run the following commands:
    python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s3TR_dr.json
    python3.7 src/VerificationScenario.py examples/scenarios_cav/qnn_s3T_dr.json

The statistics are displayed at the end in terminal after each command finishes. The important statistics of SceneChecker are: 
Nref: Number of mode-splits performed
|S|: Number of concrete modes
|S_v|^i: Number of modes in the initial abstraction
|E_v|^i: Number of edges in the initial abstraction
|S_v|^f: Number of modes in the final (after refinement) abstraction
|E_v|^f: Number of edges in the final (after refinement) abstraction
Rc: The total number of calls to computeReachset
Rt: The total time spent in reachset computation in minutes
Tt: The total computation time in minutes

Our results for the fixed random seed and averaging approaches for tables 1 and 2 are shown below.
Note that entries that we mark with NA are experiments that are also marked in the paper with NA representing cases when the corresponding tool resulted in an error or timed out (more than 2 hours on the reference machine). The entries that we mark with TO are experiments that take a total time that exceed 5 hours and may not be feasible for the reviewers to repeat:
Fixed random seed approach:
    Table1
    SceneChecker+DryVR                      CacheReach      DryVR
    sc      Nrefs   Rc      Rt      Tt      Rc      Tt      Tt
    C-S1    1       4       0.14    0.15    46      1.73    1.28
    C-S2    0       1       0.04    0.65    524     19.92   10.57
    C-S3    0       1       0.04    4.24    502     19.33   71.41
    C-S4.a  2       7       0.26    4.37    404     15.84   94.62
    C-S4.b  10      39      1.43    8.69    404     16.06   96.02
    Q-S1    1       4       0.04    0.05    NA      NA      0.25
    Q-S2    0       1       0.04    0.88    NA      NA      4.97    
    Q-S3    0       1       0.06    5.9     NA      NA      46.34
    Q-S4.a  0       1       0.06    3.17    NA      NA      56.19
    Q-S5    0       36      0.85    3.04    NA      NA      8.03

    Table2
    sc          Nref    |S|     |S_v|^i |E_v|^i |S_v|^f |E_v|^f Rc      Rt      Tt
    CNN-S2:     6       140     1       1       7       17      19      1.51    3.05
    CNN-S4:     9       520     1       1       10      28      47      3.77    11.25
    QNN-S2_TR: 	3       140     1       1       4       9       9       0.61    3.55
    QNN-S3_TR: 	5       458     1       1       6       16      15      1.51    12.7
    QNN-S4_TR: 	4       520     1       1       5       13      11      1.11    7.43
    QNN-S2_T:   0       140     7       19      7       19      8       0.53    1.38
    QNN-S3_T:   4       458     7       30      11      58      29      2.92    16.88
    QNN-S4_T:   0       520     7       30      7       30      13      1.32    5.34

Averaging approach:
    Table1
    SceneChecker+DryVR                      CacheReach      DryVR
    sc      Nrefs   Rc      Rt      Tt      Rc      Tt      Tt
    C-S1    1       4       0.14    0.15    46      1.73    1.24
    C-S2    0       1       0.04    0.65    524     20.02   9.41
    C-S3    0       1       0.04    4.21    502     19.73   TO
    C-S4.a  2       7       0.26    4.4     404     15.96   TO
    C-S4.b  10      39      1.42    8.71    404     15.83   TO 
    Q-S1    1       4       0.05    0.05    NA      NA      0.24
    Q-S2    0       1       0.04    0.86    NA      NA      4.74
    Q-S3    0       1       0.06    5.87    NA      NA      TO
    Q-S4.a  0       1       0.06    3.19    NA      NA      TO
    Q-S5    0       36      0.86    3.05    NA      NA      7.47

    Table2
    sc          Nref    |S|     |S_v|^i |E_v|^i |S_v|^f |E_v|^f Rc      Rt      Tt
    CNN-S2:     5.7     140     1       1       6.7     16.3    20.2    1.48    3.47   
    CNN-S4:     10.33   520     1       1       11.33   33.33   47.67   3.56    14.97
    QNN-S2_TR: 	4.63    140     1       1       5.63    13.88   11.88   0.85    3.72
    QNN-S3_TR: 	5.89    458     1       1       6.89    19.11   16.56   1.6     12.18
    QNN-S4_TR: 	4       520     1       1       5       13      11      1.26    8.46
    QNN-S2_T:   0       140     7       19      7       19      8.4     0.73    1.94
    QNN-S3_T:   3.67    458     7       30      10.67   56.33   28.5    2.83    18.01
    QNN-S4_T:   0       520     7       30      7       30      13      1.41    5.75

The computer that we used to do the testing has the following specifications:
32 GB RAM
Ubuntu 16.04
AMD Ryzen 7 5800X CPU @ 3.80GHz

The main file that parse the input, computes the abstraction, implements the verification and refinement algorithms is VerificationScenario.py. The file Agent.py stores the data structures of the abstraction and calls the reachability engine tools (DryVR or Flow*) to compute the reachsets. The examples folder has examples scenario files (.json files) and model files (dynamics + symmetries).

The log files of the runs of SceneChecker will be stored in the SceneChecker/log directory.

