import importlib

def importFunctions(path):
    """
    Load simulation function from given file path
    Note the folder in the examples directory must have __init__.py
    And the simulation function must be named TC_Simulate
    The function should looks like following:
        TC_Simulate(Mode, initialCondition, time_bound)

    Args:
        path (str): Similator directory.

    Returns:
        simulation function

    """
    path = path.replace('/', '.')
    try:
        module = importlib.import_module(path)
    except:
        print("Import simulation function failed on:", path)

    return [module.TC_Simulate, module.get_transform_information, module.transform_poly_to_virtual,
            module.transform_mode_to_virtual,
            module.transform_poly_from_virtual,
            module.transform_mode_from_virtual,
            module.get_virtual_mode_parameters,
            module.transform_state_from_then_to_virtual_dryvr_string,
            module.get_flowstar_parameters,
            module.get_sherlock_parameters,
            module.transform_trace_from_virtual
            ]