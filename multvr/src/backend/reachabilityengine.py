from __future__ import annotations

from .initialset import InitialSet
import numpy as np
from typing import Tuple
import scipy


class ReachabilityEngine:
    @staticmethod
    def get_reachtube_segment(initset: InitialSet, method='PW') -> np.array:
        if initset.training_traces is None:
            initset.sample_training_traces()
        ndims: int = initset.training_traces.shape[2]
        trace_len: int = initset.training_traces.shape[1]
        center_trace = initset.training_traces[0, :, :]
        num_trace = initset.training_traces.shape[0]
        if num_trace > 10:
            simulation_traces = initset.training_traces[0:10,:,:]
        else:
            simulation_traces = initset.training_traces
        x_points: np.array = initset.training_traces[0, :, 0] - initset.training_traces[0, 0, 0]
        y_points: np.array = ReachabilityEngine.all_sensitivities_calc(initset)
        points: np.array = np.zeros((ndims - 1, trace_len, 2))
        points[np.where(initset.initial_radii != 0), 0, 1] = 1.0
        points[:, :, 0] = np.reshape(x_points, (1, x_points.shape[0]))
        points[:, 1:, 1] = y_points
        if method == 'PW':
            reachtube_segment = np.zeros((trace_len-1, 2, ndims))
            normalizing_initial_set_radii: np.array = initset.initial_radii.copy()
            normalizing_initial_set_radii[np.where(normalizing_initial_set_radii == 0)] = 1.0
            df = np.zeros(center_trace.shape)
            df[:, 1:] = np.transpose(points[:, :, 1] * np.reshape(normalizing_initial_set_radii, (ndims-1, 1)))
            reachtube_segment[:, 0, :] = np.minimum(center_trace[1:, :] - df[1:, :], center_trace[:-1, :] - df[:-1, :])
            reachtube_segment[:, 1, :] = np.maximum(center_trace[1:, :] + df[1:, :], center_trace[:-1, :] + df[:-1, :])
            return reachtube_segment, simulation_traces
        else:
            print('Discrepancy computation method,', method, ', is not supported!')
            raise ValueError





    @staticmethod
    def all_sensitivities_calc(initset: InitialSet):
        num_traces: int
        trace_len: int
        ndims: int
        num_traces, trace_len, ndims = initset.training_traces.shape
        normalizing_initial_set_radii: np.array = initset.initial_radii.copy()
        points: np.array = np.zeros((normalizing_initial_set_radii.shape[0], trace_len-1))
        normalizing_initial_set_radii[np.where(normalizing_initial_set_radii == 0)] = 1.0
        normalized_initial_points: np.array = initset.training_traces[:, 0, 1:] / normalizing_initial_set_radii
        initial_distances = scipy.spatial.distance.pdist(normalized_initial_points, 'chebyshev')
        for cur_dim_ind in range(1, ndims):
            for cur_time_ind in range(1, trace_len):
                points[cur_dim_ind - 1, cur_time_ind - 1] = np.max((scipy.spatial.distance.pdist(np.reshape(initset.training_traces[:, cur_time_ind, cur_dim_ind], (initset.training_traces.shape[0], 1)), 'chebychev')
                            / normalizing_initial_set_radii[cur_dim_ind-1]) / initial_distances)
        return points



"""
Not necessary because of pdist in scipy
    @staticmethod
    def normalized_pdist(initial_points: np.array, initial_set_radii: np.array) -> np.array:
        initial_set_radii: np.array = initial_set_radii.copy()
        num_input_vecs: int = initial_points.size[0]
        num_vec_pairs: int = (num_input_vecs*(num_input_vecs-1))/2
        result: np.array = np.zeros((1, num_vec_pairs))
        ind: int = 0
        initial_set_radii[np.where(initial_set_radii == 0)] = 1.0
        for vec1_ind in range(num_input_vecs-1):
            for vec2_ind in range(vec1_ind+1, num_input_vecs):
                result[1, ind] = np.max((initial_points[vec1_ind, :] - initial_points[vec2_ind, :]))
                ind += 1
        return result  
"""