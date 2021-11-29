from typing import Tuple, Optional, DefaultDict, Dict, List, Iterable
from collections import deque, defaultdict
import numpy as np

# returns partitions of an Initial set to be verified
def get_refinements(init_set: 'Initial_Set') -> Iterable['Initial_Set']:
    # adds 1 to the height of the input initial set when producing refinements
    # TODO determine refinement strategy

# computes and returns collection of the successor initial sets
def get_next_initsets(cur_RT_segment: np.array, all_guards: List['Guard']) -> Iterable['InitialSet']:
    result = []
    for guard in all_guards:
        result.append(get_next_initset(cur_RT_segment, guard))
    pass



"""
Inputs: All originally safe reachtube segements already computed for the current mode and their computed next initial sets  
Functionality: checks the safety of each element of the input
Output: Any initial sets in the current mode that correspond to any of the now unsafe Input elements
"""
def get_unsafe_and_unknown_initial_sets():
    pass


def build_safe_reachtube(agent, cur_scenario) -> Tuple[str, 'Scenario']:
    # initialize the state space exploration to begin at the initial mode and initial set.
    mode_sequence_stack: deque = deque(((cur_scenario.initialModeInd,),))
    cur_scenario.remaining_initial_sets: DefaultDict[Tuple[int], deque] = defaultdict(deque)
    cur_scenario.remaining_initial_sets[(cur_scenario.initialModeInd,)].append((cur_scenario.initial_set,))
    cur_scenario.RT: DefaultDict[Tuple[int], List[ReachtubeTreeNode]] = defaultdict(list) # Stores all the finalized RT segments in a forest for each mode sequence
    while len(mode_sequence_stack) > 0:
        refine_parent_flag = False
        cur_mode_sequence: Tuple[int] = mode_sequence_stack.pop()
        cur_mode_stack: deque = cur_scenario.remaining_initial_sets[cur_mode_sequence]
        # Cache the guard intersections here (list of next initial sets) so that we can use them for safety checking if we go back to parent mode.
        for init_set in get_unsafe_and_unknown_initial_sets(cur_mode_sequence, cur_scenario.RT, cur_scenario):  # checks if formerly safe Reachtubes in
            # this mode are still safe and iterates over initial set partitions (of leaf nodes) of the now unsafe tube segments
            if at_refinement_limit():
                # go to the parent mode and refine there
                # consider storing trajectories to also verify their safety (probably too much memory though)
                cur_scenario.add_to_unknown_set(init_set) # unknown
                if not refine_parent_flag:
                    mode_sequence_stack.append(cur_mode_sequence)
                    mode_sequence_stack.append(cur_mode_sequence[:-1])
                refine_parent_flag = True
            else:
                cur_mode_stack.extend(get_refinements(init_set))  # supports arbitrary refinement strategy & extend is repeated appends
        if refine_parent_flag:
            continue
        while len(cur_mode_stack) > 0:
            cur_initial_set = cur_mode_stack.pop()
            cur_traces: np.array = cur_initial_set.sample_traces()
            unknown_trace_inds: np.array, unsafe_trace_inds:  np.array = extract_unsafe_and_unknown_traces(traces)
            if len(unknown_trace_inds) > 0 or len(unsafe_trace_inds) > 0:
                if len(cur_mode_sequence) == 1 and len(unsafe_trace_inds) > 0:  # Unsafe Trace found,
                    # return the counterexample to the user.
                    return False, get_complete_unsafe_trajectory()  # TODO don't just return the unsafe trace in
                    # the root mode, but concatenate it across all modes (will need to store a mapping from dynamic
                    # unsafe set to trajectory that caused the corresponding dynamic unsafe set to be added).
                elif len(cur_mode_sequence) == 1:  # trace that hits unknown set found in first mode
                    # Return unknown, and probably useful to return both the verified portions and their corresponding
                    # RTs as well as return the unverified initial sets. (speculation)
                    return None, get_desired_unknown_return()
                if len(unsafe_trace_inds) > 0: # add the
                    cur_scenario.add_traces_to_dynamic_unsafe_set(traces, unsafe_trace_inds)
                if len(unknown_trace_inds) > 0:
                    cur_scenario.add_traces_to_unknown_set(traces, unknown_trace_inds)
                mode_sequence_stack.append(cur_mode_sequence)
                mode_sequence_stack.append(cur_mode_sequence[:-1])
                break
            # else case
            cur_RT_segment: np.array = compute_RT_segment(cur_initial_set, traces)
            next_initsets: Iterable['InitialSet'] = get_next_initsets(cur_RT_segment, all_guards)
            if not is_all_safe(cur_RT_segment, next_initsets):  # check safety of current Reachtube and next init sets
                if at_refinement_limit():  # need to go to parent mode and refine the parent's initial set
                    cur_scenario.add_to_dynamic_unknown_set(cur_initial_set)
                    mode_sequence_stack.append(cur_mode_sequence)
                    mode_sequence_stack.append(cur_mode_sequence[:-1])
                    break
                else:
                    # refine the current mode's initial set Still need to create empty node to be the parent for the children in the tree.
                    new_node = ReachtubeTreeNode(cur_initial_set.height, cur_initial_set, None, cur_initial_set.parent)
                    if cur_intial_set.parent is not None:
                        cur_initial_set.parent.children.append(new_node)
                    else:
                        cur_scenario.RT[cur_mode_sequence].append(new_node)  # add a new tree to the RT forest
                    cur_mode_stack.extend(get_refinements(cur_initial_set)) # get refined initial sets.
            else:
                # reachtube is safe all is well, add it to the tree, and make successor initial sets to the frontier.
                new_node = ReachtubeTreeNode(cur_initial_set.height, cur_initial_set, cur_RT_segment, next_initsets, cur_initial_set.parent)
                if cur_intial_set.parent is not None:
                    cur_initial_set.parent.children.append(new_node)  # set this as a child node for the parent.
                else:
                    cur_scenario.RT[cur_mode_sequence].append(new_node)  # add a new tree to the RT forest
                # add guard intersections to frontier
                for init_set in next_initsets:
                    # TODO, maybe more efficiently merge existing initial sets with this one.
                    cur_scenario.remaining_initial_sets[cur_mode_sequence + init_set.mode_ind].append(init_set)
                # add successor mode sequences to the mode sequence stack
                cur_scenario.mode_sequence_stack.extend(cur_mode_sequence + val for val in set(map(lambda x: x.mode_ind, next_initsets)))
    # whole scenario is safe, we are done, and we can return the RT
    return True, cur_scenario.RT




def verify(cur_scenario: 'Scenario') -> Tuple[Optional[bool], 'Result']:
    refinement_limit = 10 # TODO make this a configuration parameter
    for agent in Scenario.agents_list:
        result = build_safe_reachtube(agent, Scenario)
        if result[0] == 'Unknown':
            print(f"refinement limit reached: {refinement_limit}")
            return (False, result[1])
        elif result[0] == 'False':
            print(f"Unsafe behavior found!")
            return (None, result[1])
        elif result[0] != 'True':
            raise ValueError("invalid string returned by build_safe_reachtube")
    return (True, Scenario.certificate)

