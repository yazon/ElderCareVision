"""Placeholder for the Person State Analyzer agent."""

import logging

from elder_care_vision.core.agents.base_agent import BaseAgent
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class PersonState(BaseModel):
    fall_confidence: int
    """Represents the confidence of a fall."""


class PersonStateAnalyzerAgent(BaseAgent):  # Inherit from BaseAgent
    """Analyzes the state of a person based on available data, using BaseAgent."""

    PROMPT = """
    You are a person state analyzer agent. TODO: Implement actual analysis logic.
    """

    def __init__(self) -> None:
        """Initializes the Person State Analyzer agent using the BaseAgent."""
        logger.info("Initializing Person State Analyzer Agent (via BaseAgent)")
        # Call the BaseAgent's __init__ with the specific agent name for config loading
        super().__init__(agent_name="person_state_analyzer", output_type=PersonState)
