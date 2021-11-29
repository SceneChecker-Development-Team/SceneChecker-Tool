from typing import Callable, Tuple, List, Dict
import numpy as np
from .model import Model
from multvr.src.backend.initialset import InitialSet
import json



class Scenario:
    def __init__(self, scenario_file_name: str):
        self.read_input_file(scenario_file_name)
        """
        # TODO implement this stuff
        self.models_dynamics: List[Model] = []
        for model in inputData["models"]:
            if inputData[model]["type"] == "black_box":
                self.models_dynamics.append(importSimFunctions(model["folder"]))
            elif inputData[model]["type"] == "c2e2":
                self.models_dynamics.append(parseC2E2Model(model["folder"]))
            else:
                raise ValueError(f"invalid model definition for Model #{ind}")
        agent_automatas = list(map(lambda x: parseAutomata(data[x]), data["agents"]))
        agent_unsafe_sets: List[Dict[str, 'StateSet']] = [parseUnsafeSet(data[agent]) for agent in data["agents"]]
        """
        return

    def read_input_file(self, scenario_file_name):
        assert ".json" in scenario_file_name, "Please provide json input file with .json file extension!"
        input_data: Dict = {}
        try:
            with open(scenario_file_name) as scenario_file:
                input_data = json.load(scenario_file)
        except ValueError:
            print("Value Error")
            raise ValueError
        except:
            print("Invalid or unreadable json file:", scenario_file_name)
            raise ValueError
        if 'legacy' not in input_data:
            input_data['legacy'] = True
        if input_data['legacy']:
            #  parse a single hybrid system from the file
            # TODO: support more than single mode systems on parse
            # Error on legacy required input values
            if 'vertex' not in input_data:
                raise ValueError
            if 'variables' not in input_data:
                raise ValueError
            if 'initialSet' not in input_data:
                raise ValueError
            else:
                input_data['initialSet'] = np.array(input_data['initialSet'])
            if 'timeHorizon' not in input_data:
                raise ValueError
            if 'directory' not in input_data:
                raise ValueError
            if (2, len(input_data['variables'])) != input_data['initialSet'].shape:
                raise ValueError
            # Error if exercising a feature that has yet to be implemented
            if len(input_data['vertex']) != 1:
                raise NotImplemented
            else:
                initial_mode: str = input_data['vertex'][0]
            if 'edge' not in input_data:
                input_data['edge'] = []
            if 'guard' not in input_data:
                input_data['guard'] = []
            if len(input_data['edge']) > 0:
                raise NotImplemented
            if len(input_data['guard']) > 0:
                raise NotImplemented
            if 'unsafeSet' in input_data:
                raise NotImplemented

            self.scene_type: str = 'Single Agent'
            self.model: Model = Model("quadrotor", input_data['variables'], input_data['directory'])
            self.initial_set: InitialSet = InitialSet(self.model, initial_mode, input_data['initialSet'], input_data['timeHorizon'], 100)
            self.initial_set.seed = input_data['seed']

        else:
            raise NotImplemented

