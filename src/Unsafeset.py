import numpy as np
import polytope as pc
from typing import List, Tuple
import polytope as pc


class Unsafeset:

    def __init__(self, mode_id: int, unsafe_set: pc.Region):
        self.mode_id: int = mode_id
        self.unsafe_set = unsafe_set
