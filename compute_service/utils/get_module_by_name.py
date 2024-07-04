import importlib


def get_module(module_path, class_name):
    try:
        # Get the class from the module
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except Exception:
        raise ModuleNotFoundError(f"Class '{class_name}' not found in module translator.{module_path}")
    