import logging

from agents import function_tool

logger = logging.getLogger(__name__)


@function_tool
def emergency_call_tool() -> None:
    """
    Simulates placing an emergency call.

    This function currently only logs the action.
    """
    logger.info("Emergency Call tool called")
    logger.warning("Initiating emergency call (simulation).")
