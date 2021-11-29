from typing import List, Optional, Iterable
import numpy as np
from __future__ import annotations

from .initialset import InitialSet

class ReachtubeTreeNode:
    # TODO store next initsets within each node
    def __init__(self, height: int, initial_set: InitialSet, reachtube_segment: Optional[np.array],
                 parent: Optional[ReachtubeTreeNode], children: Optional[List[ReachtubeTreeNode]]=None,
                 next_initial_sets: Optional[Iterable[InitialSet]]=None):
        self.height: int = height
        self.reachtube_segment: Optional[np.array] = reachtube_segment
        self.initial_set: InitialSet = initial_set
        self.next_initial_sets: Iterable[InitialSet] = next_initial_sets
        self.parent: Optional[ReachtubeTreeNode] = parent
        # Children is a list to support arbitrary partition schema, list of partitions
        if children is None:
            self.children: List[ReachtubeTreeNode] = []
        else:
            self.children: List[ReachtubeTreeNode] = list(children)