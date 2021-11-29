from __future__ import annotations
import polytope as pc
import numpy as np
from typing import Optional, Union
import polytope as pc
from multvr.src.frontend.model import Model

class InitialSet:
    def __init__(self, model: Model, initial_mode: str, initial_shape: Optional[Union[pc.Polytope, pc.Region, np.array]], remaining_time: float, num_training_traces: Optional[int] = None):
        self.training_traces: Optional[np.ndarray] = None
        self.model: Model = model
        self.initial_mode: str = initial_mode
        self.wrap_initial_shape(initial_shape)
        self.remaining_time: float = remaining_time
        self.seed = 0
        if num_training_traces is None:
            raise NotImplemented
        else:
            self.num_training_traces = num_training_traces
        return

    def wrap_initial_shape(self, continuous_set: Optional[Union[pc.Polytope, pc.Region, np.array]]) -> np.array:
        if type(continuous_set) != np.ndarray:
            print(type(continuous_set))
            raise NotImplemented
        elif continuous_set.shape[0] != 2 or len(continuous_set.shape) != 2:
            raise ValueError
        else:
            self.initial_set_wrapper: np.array = np.copy(continuous_set)
        self.initial_radii: np.ndarray = (self.initial_set_wrapper[1, :] - self.initial_set_wrapper[0, :])/2
        self.num_dims: int = self.initial_radii.shape[0]
        self.initial_center: np.ndarray = self.initial_set_wrapper[0, :] + self.initial_radii
        return


    def sample_training_traces(self) -> None:
        assert self.num_training_traces >= 2
        if self.seed is not None:
            np.random.seed(self.seed)
        self.center_trace: np.ndarray = self.model.TC_Simulate(self.initial_mode, self.initial_center, self.remaining_time)
        self.initial_states = np.zeros((self.num_training_traces, self.num_dims))
        zero_centered_hrect_samples = ((np.random.rand(self.num_training_traces - 1, self.num_dims) - 0.5) * 2) * self.initial_radii
        self.initial_states[1:, :] = self.initial_center + zero_centered_hrect_samples
        self.training_traces = np.zeros((self.num_training_traces, self.center_trace.shape[0], self.num_dims+1))
        self.training_traces[0, :, :] = self.center_trace
        self.training_traces[1:, :, :] = self.model.TC_Simulate_Batch(self.initial_mode, self.initial_states[1:, :], self.remaining_time)
        return
