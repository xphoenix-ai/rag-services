import yaml
import torch


def sanitize_params(params: dict) -> dict:
    """Recursively evaluate string values in configuration that represent Python expressions.

    Args:
        params (dict): Dictionary of configuration parameters

    Returns:
        dict: Dictionary with string values evaluated where possible
    """
    params_copy = dict(params)
    for key, val in params_copy.items():
        if isinstance(val, str):
            try:
                val = eval(val)     # Try to evaluate all strings
                params[key] = val
            except Exception:   # String cannot be evaluated as an expression
                pass
        elif isinstance(val, dict):
            params[key] = sanitize_params(val)
    
    return params
            

def get_config(config_path: str) -> dict:
    """Read and parse YAML configuration file.

    Args:
        config_path (str): Path to YAML configuration file

    Returns:
        dict: Configuration dictionary with evaluated parameters
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return sanitize_params(config)
