import asyncio
import json
import logging
import threading
from dataclasses import asdict, is_dataclass
from enum import Enum

# Import Response from flask
from flask import Flask, Response, render_template

from elder_care_vision.core.agents.psa_data import FallDetectionResult
from elder_care_vision.core.coordinator import Coordinator, CoordinatorState
from elder_care_vision.utils.utils import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the coordinator instance
# NOTE: Accessing shared mutable state between threads requires careful consideration
# for thread safety in a production environment (e.g., using locks or queues).
# For simplicity here, we access the context directly, assuming reads are frequent
# and writes don't cause major inconsistencies during the brief read time.
coordinator: Coordinator | None = None
coordinator_thread: threading.Thread | None = None

app = Flask(__name__)

config = load_config()
# Default polling interval in seconds if not specified in config
DEFAULT_POLLING_INTERVAL_S = 0.5
polling_interval_ms = int(config.get("web", {}).get("poll_interval_s", DEFAULT_POLLING_INTERVAL_S) * 1000)


# Custom JSON encoder to handle Enum and Dataclasses
class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Enum types and dataclasses."""

    def default(self, o):
        if isinstance(o, Enum):
            return o.name  # Convert Enum members to their names
        if is_dataclass(o):
            return asdict(o)  # Convert dataclasses to dicts
        try:
            return super().default(o)
        except TypeError:
            # Attempt to convert unknown types to string as a fallback
            logger.warning(f"Could not serialize object of type {type(o).__name__}, converting to string.")
            return str(o)


# Note: Setting app.json_encoder doesn't seem sufficient for nested objects within jsonify
# app.json_encoder = CustomEncoder


def run_coordinator_in_background() -> None:
    """Runs the coordinator's async loop in a separate thread."""
    global coordinator
    logger.info("Coordinator background thread started.")
    coordinator = Coordinator()
    try:
        # Ensure an event loop exists for the thread
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
        loop.run_until_complete(coordinator.run())
    except Exception as e:
        logger.exception(f"Coordinator thread encountered an error: {e}")
    finally:
        logger.info("Coordinator background thread finished.")


@app.route("/")
def index() -> str:
    """Serves the main HTML page."""
    # Pass the polling interval to the template
    return render_template("index.html", polling_interval_ms=polling_interval_ms)


@app.route("/api/context")
def get_context() -> Response:
    """Provides the current coordinator context as JSON."""
    global coordinator
    if coordinator and coordinator.context:
        # Explicitly dump the object using the custom encoder
        json_data = json.dumps(coordinator.context, cls=CustomEncoder, indent=None)  # No indent for API response
        return Response(json_data, mimetype="application/json")
    # Return a default/empty state if coordinator is not ready
    default_context = {
        "current_state": CoordinatorState.ANALYZING_IMAGE.name,
        "last_psa_confidence": 0,
        "health_status": None,
        "fall_detection_result": FallDetectionResult(),  # Pass the dataclass instance
    }
    # Explicitly dump the default context using the custom encoder
    json_data = json.dumps(default_context, cls=CustomEncoder, indent=None)
    return Response(json_data, mimetype="application/json")


def start_background_tasks() -> None:
    """Starts the background thread for the coordinator."""
    global coordinator_thread
    if coordinator_thread is None or not coordinator_thread.is_alive():
        logger.info("Starting coordinator background thread.")
        coordinator_thread = threading.Thread(target=run_coordinator_in_background, daemon=True)
        coordinator_thread.start()


if __name__ == "__main__":
    start_background_tasks()
    # Note: Using Flask's development server is not recommended for production.
    # Use a production-ready WSGI server like Gunicorn or Waitress.
    app.run(debug=True, host="0.0.0.0", port=5001, use_reloader=False)  # use_reloader=False is important for threads
