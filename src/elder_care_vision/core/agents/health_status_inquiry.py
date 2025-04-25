"""Placeholder for the Health Status Inquiry agent."""

import logging

from .base_agent import BaseAgent
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class HealthStatus(BaseModel):
    health_status: str
    """Represents the health status of a person."""


class HealthStatusInquiryAgent(BaseAgent):
    """Conducts inquiries about a person's health status, using BaseAgent."""

    PROMPT = """
    You are a health status inquiry agent. TODO: Implement actual inquiry logic.
    """

    def __init__(self) -> None:
        """Initializes the Health Status Inquiry agent using the BaseAgent."""
        logger.info("Initializing Health Status Inquiry Agent (via BaseAgent)")
        # Call the BaseAgent's __init__ with the specific agent name for config loading
        super().__init__(agent_name="health_status_inquiry", output_type=HealthStatus)
