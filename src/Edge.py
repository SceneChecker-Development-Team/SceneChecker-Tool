import numpy as np
import polytope as pc
from typing import List
from src.Waypoint import Waypoint


class Edge:

    def __init__(self, mode1: int, mode2: int, id: int,
                 guard, region_guard: pc.Region = pc.Region(list_poly=[])):
        self.source: int = mode1
        self.dest: int = mode2
        self.guard = guard
        self.region_guard: pc.Region = region_guard
        self.id = id
        # self.delta: np.array = (self.guard[1, :] - self.original_guard[0, :]) / 2
    # TODO add helper function to check if point is inside guard

    def is_equal(self, other_edge: List[float]):
        return self.source == other_edge.source and self.dest == other_edge.dest
