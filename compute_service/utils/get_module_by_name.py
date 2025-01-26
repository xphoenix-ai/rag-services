import importlib


def get_module(module_path: str, class_name: str) -> type:
    """Dynamically import and return a class from a module.

    Args:
        module_path (str): Dot-separated path to the module
        class_name (str): Name of the class to import

    Returns:
        type: The requested class

    Raises:
        ModuleNotFoundError: If the module or class cannot be found
    """
    try:
        # Get the class from the module
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception:
        raise ModuleNotFoundError(f"Class '{class_name}' not found in module {module_path}")
    