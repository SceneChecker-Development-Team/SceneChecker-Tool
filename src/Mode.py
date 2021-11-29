import numpy as np
import polytope as pc
from typing import List


class Mode:

    edge: pc.Polytope

    def __init__(self, given_edge: Edge):
        self.edge = given_edge
    # TODO add helper function to check if point is inside guard
