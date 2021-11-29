import polytope as pc
from typing import List, Tuple
import numpy as np
from src.Waypoint import Waypoint

from src.PolyUtils import PolyUtils

ABS_TOL = 0.000001


class ReachtubeSegment:

    # tube is a list of polytopes without time
    # time is index zero for every point in simulation, followed by state variables
    # discrepency function will just be DryVR style right now
    '''
    def __init__(self, tube: List[pc.Region], trace: np.array, system_origin: np.array, system_angle: float,
                 is_unified: bool, guard_min_index: int, guard_max_index: int, next_initset: pc.Region, virtual_mode: Tuple[float, ...]):
    '''
    def __init__(self, tube_list: List[List[pc.Region]], tube_list_rect: List[List[np.array]], trace: np.array, guard_min_index: int, guard_max_index: int,
                 next_initset: pc.Region,
                 virtual_mode: int):
        self.tube_list: List[List[pc.Region]] = tube_list
        self.trace: np.array = trace
        #self.system_origin: np.array = system_origin
        #self.system_angle: float = system_angle
        #self.is_unified: bool = is_unified
        self.guard_min_index: int = guard_min_index
        self.guard_max_index: int = guard_max_index
        self.next_initset: pc.Region = next_initset
        self.virtual_mode: int = virtual_mode
        self.tube_list_rect: List[List[np.array]] = tube_list_rect

    # TODO union function between Reachtubes

    # TODO Coordinate Transform function

    def is_poly_intersect(self, poly: pc.Polytope):
        for tube in self.tube_list:
            for tube_poly in tube:
                if not pc.is_empty(tube_poly.intersect(poly, abs_tol=ABS_TOL)):
                    return True
        return False

    # TODO check Reachtube intersection function
    # we don't need it for now
    # def intersect(self, tube2):


if __name__ == '__main__':
    pass
    # TODO write self tests and plotting
