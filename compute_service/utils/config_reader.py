import yaml
import torch


def sanitize_params(params):
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
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return sanitize_params(config)
