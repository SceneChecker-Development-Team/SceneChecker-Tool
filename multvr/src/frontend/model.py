import os
import sys
import importlib
from typing import List

class Model:
    def __init__(self, name: str, variable_names: List[str], directory_path: str):
        self.name: str = name
        self.importModelFunctions(directory_path)
        return

    def importModelFunctions(self, path: str) -> None:
        """
        Load simulation function from given file path
        Note the folder in the examples directory must have __init__.py
        And the simulation function must be named TC_Simulate
        The function should looks like following:
            TC_Simulate(Mode, initialCondition, time_bound)

        Args:
            path (str): Simulator directory.

        Effect:
            adds simulation functions to class

        """
        sys.path.append(os.path.abspath(path))
        mod_name = path.replace('/', '.')
        module = importlib.import_module(mod_name)
        sys.path.pop()
        try:
            self.TC_Simulate_Batch = module.TC_Simulate_Batch
        except AttributeError:
            self.TC_Simulate_Batch = None
        try:
            self.TC_Simulate = module.TC_Simulate
        except AttributeError:
            self.TC_Simulate = None
        return



if __name__ == '__main__':
    my_test = Model("lorentz", ["x", "y", "z"], "examples/lorentz")
    assert my_test.TC_Simulate_Batch is not None
