"""Placeholder for the Health Status Inquiry agent."""

import logging

from agents import function_tool
from pydantic import BaseModel

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)


class HealthStatus(BaseModel):
    health_status: str
    """Represents the health status of a person."""


@function_tool
def health_status_inquiry_tool(msg: str) -> str:
    """Inquires about a person's health status."""
    logger.info(f"Health status inquiry tool called with message: {msg}")
    return "I am fine!"


class HealthStatusInquiryAgent(BaseAgent):
    """Conducts inquiries about a person's health status, using BaseAgent."""

    PROMPT = """
    You are a health status inquiry agent. Ask user if everything is okay.
    """

    def __init__(self) -> None:
        """Initializes the Health Status Inquiry agent using the BaseAgent."""
        logger.info("Initializing Health Status Inquiry Agent (via BaseAgent)")
        # Call the BaseAgent's __init__ with the specific agent name for config loading
        super().__init__(agent_name="health_status_inquiry", output_type=HealthStatus)
        self.agent.tools.append(health_status_inquiry_tool)
