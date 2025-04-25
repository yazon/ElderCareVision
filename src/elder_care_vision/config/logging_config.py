import logging
import logging.config
import logging.handlers
from datetime import UTC, datetime
from pathlib import Path

import yaml


def setup_logging(
    default_level: int = logging.INFO,
) -> None:
    """
    Set up logging configuration from a YAML file or environment variable.

    Ensures the log directory exists and configures logging using dictConfig.
    Allows overriding the config file path via an environment variable.
    Dynamically adds a timestamp to the log filename defined in the YAML config.

    Args:
        config_path: Path to the logging configuration file (YAML).
        default_level: Default logging level if config file is not found.
        env_key: Environment variable name to check for config file path override.
    """
    config_path = Path(__file__).parent / "logging.yaml"

    project_root = Path(__file__).parent.parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if config_path.is_file():
        try:
            with Path.open(config_path) as f:
                config_data = yaml.safe_load(f.read())

            # Dynamically update the log filename to include a timestamp
            if "file" in config_data.get("handlers", {}):
                handler_config = config_data["handlers"]["file"]
                base_filename = handler_config.get("filename", "app.log")

                # Ensure filename path is treated correctly
                file_path = Path(base_filename)
                timestamp = f"{datetime.now(UTC):%Y-%m-%d_%H-%M-%S}"
                # Place timestamped log inside the log_dir
                log_filename = log_dir / f"{file_path.stem}_{timestamp}{file_path.suffix}"

                # Update the filename in the config dictionary
                handler_config["filename"] = str(log_filename.resolve())

            logging.config.dictConfig(config_data)
        except Exception as e:
            logging.basicConfig(
                level=default_level,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )
            logging.exception(  # noqa: LOG015
                "Error loading logging configuration from %s. Using basicConfig.",
                config_path,
                exc_info=e,
            )
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logging.warning(  # noqa: LOG015
            "Logging configuration file not found at %s. Using basicConfig.",
            config_path,
        )
