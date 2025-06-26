import yaml
from typing import Any, Dict


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary.

    Args:
        path: The file path to the YAML configuration file.

    Returns:
        A dictionary representing the parsed YAML content.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If the YAML file has invalid syntax.

    Example:
        config = load_yaml_config("config.yaml")
        print(config['vehicle_config']['max_speed'])
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def print_config(config: Dict[str, Any]):
    """Pretty-print a configuration dictionary to the console.

    Args:
        config: The configuration dictionary to print.

    Example:
        config = load_yaml_config("config.yaml")
        print_config(config)
    """
    import pprint
    pprint.pprint(config)
