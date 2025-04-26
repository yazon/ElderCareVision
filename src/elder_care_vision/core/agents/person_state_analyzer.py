"""Placeholder for the Person State Analyzer agent."""

import logging

from agents import function_tool
from pydantic import BaseModel

from elder_care_vision.core.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


@function_tool
def analyze_image(image_data: str) -> str:
    """Analyze the image and return a string describing the person's state."""
    logger.info(f"Analyzing image: {image_data}")
    return "Person is laying on the floor"


class PersonState(BaseModel):
    fall_confidence: int
    """Represents the confidence of a fall."""


class PersonStateAnalyzerAgent(BaseAgent):  # Inherit from BaseAgent
    """Analyzes the state of a person based on available data, using BaseAgent."""

    PROMPT = """
    You are a person state analyzer agent. Return random number between 0 and 100 as confidence of a fall.
    """
    handoffs = []

    def __init__(self) -> None:
        """Initializes the Person State Analyzer agent using the BaseAgent."""
        logger.info("Initializing Person State Analyzer Agent (via BaseAgent)")
        super().__init__(agent_name="person_state_analyzer", output_type=PersonState)

        self.agent.tools.append(analyze_image)

        logger.info(f"Person State Analyzer Agent initialized with model: {self.model}")
