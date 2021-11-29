import numpy as np
import polytope as pc
from typing import List, Tuple
from src.Unsafeset import Unsafeset


class Waypoint:

    def __init__(self, mode: str, mode_parameters: List[float], time_bound: float, id: int,
                 unsafeset_list: List[Unsafeset] = None):
        self.mode: str = mode
        self.mode_parameters: List[float] = mode_parameters
        self.time_bound: float = time_bound
        self.id = id
        self.unsafeset_list = unsafeset_list

    def is_equal(self, other_waypoint: List[float]):
        return tuple(self.mode_parameters) == tuple(other_waypoint.mode_parameters)
        # self.delta: np.array = (self.original_guard[1, :] - self.original_guard[0, :]) / 2
    # TODO add helper function to check if point is inside guard
