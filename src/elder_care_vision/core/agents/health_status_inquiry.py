"""Placeholder for the Health Status Inquiry agent."""

import logging

from agents import function_tool
from pydantic import BaseModel

from .base_agent import BaseAgent

# Import the new voice workflow execution function (assuming it's placed appropriately)
from .health_status_voice_workflow import HealthStatusResult, run_health_inquiry_voice_interaction

logger = logging.getLogger(__name__)


class HealthStatus(BaseModel):
    health_status: str
    """Represents the health status of a person."""


@function_tool
def health_status_inquiry_tool(msg: str) -> str:
    """Inquires about a person's health status. (Text-based, kept for now)"""
    logger.info(f"Health status inquiry tool called with message: {msg}")
    # This tool is text-based, the voice interaction is separate
    return "I am fine! (Text Tool Response)"


class HealthStatusInquiryAgent(BaseAgent):
    """
    Conducts inquiries about a person's health status.

    This agent originally used text-based interaction via BaseAgent.
    The voice-based workflow is now handled by `run_health_inquiry_voice_interaction`.
    This class might coordinate the overall process or be refactored depending
    on how the voice interaction is integrated into the larger application.
    """

    PROMPT = """
    You are a health status inquiry agent. Your primary interaction method
    is now voice-based, handled externally. This prompt is less relevant
    for the direct voice flow but kept for context or potential text fallbacks.
    """

    def __init__(self) -> None:
        """Initializes the Health Status Inquiry agent using the BaseAgent."""
        logger.info("Initializing Health Status Inquiry Agent (via BaseAgent)")
        # Call the BaseAgent's __init__ with the specific agent name for config loading
        super().__init__(agent_name="health_status_inquiry", output_type=HealthStatus)
        # The voice interaction does not directly use this agent's tools.
        # self.agent.tools.append(health_status_inquiry_tool) # Might remove later

    async def run_voice_check(self) -> HealthStatusResult:
        """
        Initiates and runs the voice-based health status check.

        Returns:
            HealthStatusResult: The result of the voice interaction.
        """
        logger.info("Running voice-based health check...")
        # Import here to avoid circular dependency if workflow uses agent parts later
        result = await run_health_inquiry_voice_interaction()
        logger.info(f"Voice check completed with result: {result}")
        return result

    # TODO: Decide how the result of run_voice_check integrates with the rest
    # of the application logic. Does it update state? Trigger other agents?
