from math import exp
from scipy.optimize import linprog
import numpy as np
from itertools import combinations
from typing import Optional
from src.Waypoint import Waypoint
import pdb

class DiscrepancyLearning:

    @staticmethod
    def get_random_trace_starts(initial_set_box: np.array, num_samples):
        starts = np.random.ranf((num_samples, initial_set_box.shape[1]))
        return starts * (initial_set_box[1, :] - initial_set_box[0, :]) + initial_set_box[0, :]

    @staticmethod
    def sample_traces(initial_set_box: np.array, waypoint: Waypoint, time_step: int, time_bound: '',
                            cur_agent: 'Agent', num_samples, b_inds, max_iters=100000):
        dim = initial_set_box.shape[1]
        trace_len = int(np.ceil((time_bound + time_step) / time_step))
        # TAC uncomment below
        if  False:#num_samples > 0:
            for j in range(max_iters):
                points = np.row_stack((np.average(initial_set_box, axis=0),
                                       DiscrepancyLearning.get_random_trace_starts(initial_set_box, num_samples)))
                if np.min(np.abs(points[b_inds[0, :], :] - points[b_inds[1, :], :]), axis=None) > 0.01:
                    traces = np.empty((num_samples + 1, trace_len, dim))
                    for i in range(num_samples):
                        traces[i, :, :] = cur_agent.simulate_segment(points[i, :], waypoint, time_step, time_bound)[:, 1:]
                    return traces, trace_len
        #print("Trace sampling iteration limit reached")
        return cur_agent.simulate_segment(np.average(initial_set_box, axis=0), waypoint, time_step, time_bound)[:, 1:], False

    @staticmethod
    def compute_k_and_gamma(initial_set_box: np.array, waypoint: Waypoint, time_step: int, time_bound: float,
                            cur_agent: 'Agent'):
        dim: int = initial_set_box.shape[1]
        num_traces: int = 2
        traces: np.array
        trace_len: int
        if num_traces > 0:
            b_inds = np.column_stack(tuple(combinations(range(num_traces), 2)))
        else:
            b_inds = None
        traces, trace_len = DiscrepancyLearning.sample_traces(initial_set_box, waypoint, time_step, time_bound, cur_agent, num_traces, b_inds)
        if trace_len:
            center_trace = traces[0, :, :]

        else:
            center_trace = traces
        return np.ones((dim,)), np.zeros((dim,)), center_trace

        n: float = 100.0
        C = [n, (n + 1) / 2 * (trace_len)]
        A = -1 * np.ones((trace_len - 1, 2))
        A[:, 1] = center_trace[1:, 0] - center_trace[0, 0]
        assert num_traces >= 3, "More traces than 3 required"
        B = np.min(np.tile(np.reshape(np.log(np.abs(traces[b_inds[0, :], 0, :] - traces[b_inds[1, :], 0, :])),
                                      (b_inds.shape[1], 1, traces.shape[2])), (1, trace_len-1, 1))
                   - np.log(np.clip(np.abs(traces[b_inds[0, :], 1:, :] - traces[b_inds[1, :], 1:, :]), 0.001, None)), axis=0)
        k_bounds = (0, None)  # k >= 1, i.e. log(k) >= 0
        gamma_bounds = (None, None)
        k = []
        gamma = []
        for i in range(B.shape[1]):
            res = linprog(C, A_ub=A, b_ub=B[:, i], bounds=(k_bounds, gamma_bounds))
            print(res.x[0])
            try:
                my_res = exp(res.x[0])
            except OverflowError:
                print("Using fallback")
                k.append(1)
                gamma.append(0)
                continue
            k.append(my_res)
            gamma.append(res.x[1])
            # use glpk
        """
            lp = glpk.LPX()
            lp.name = 'logk_gamma'
            lp.obj.maximize = False  # set this as a minimization problem
            lp.rows.add(len(A))  # append rows to this instance
            for i in range(len(b)):
                lp.rows[i].bounds = None, b[i]  # set bound: entry <= b[i]
            lp.cols.add(2)  # append two columns for k and gamma to this instance
            lp.cols[0].name = 'logk'
            lp.cols[0].bounds = 0.01, 10.0  # k >= 1, i.e. log(k) >= 0
            lp.cols[1].name = 'gamma'
            lp.cols[1].bounds = None, None  # no constraints for gamma
            lp.obj[:] = c  # set objective coefficients
            lp.matrix = np.ravel(A)  # set constraint matrix; convert A to 1-d array
            lp.simplex()  # solve this LP with the simplex method
            k = exp(lp.cols[0].primal)
            gamma = lp.cols[1].primal
        """
        return np.array(k), np.array(gamma), traces[0, :, :]

