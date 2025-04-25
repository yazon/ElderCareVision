import json
from pathlib import Path


class UtilsError(Exception):
    """Exception for utils errors."""


def load_config() -> dict[str, any]:
    """
    Load configuration from config.json file.

    Returns:
        Dict[str, Any]: Configuration dictionary

    """
    config_path = Path(__file__).parent.parent / "config/config.json"
    try:
        with Path.open(config_path) as f:
            return json.load(f)
    except FileNotFoundError as e:
        err = f"Config file not found at {config_path}"
        raise UtilsError(err) from e
    except json.JSONDecodeError as e:
        err = f"Invalid JSON in config file: {e}"
        raise UtilsError(err) from e


def get_static_path() -> Path:
    """
    Get the absolute path to the static data directory.

    Returns:
        Path: Absolute path to the static data directory
    """
    # The static directory is located at the same level as the config directory
    return Path(__file__).parent.parent / "static"
