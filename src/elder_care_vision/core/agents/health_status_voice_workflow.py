"""Voice workflow for the Health Status Inquiry agent using OpenAI agents SDK."""

import logging
from collections.abc import AsyncIterator

from agents import Agent
from agents.voice import VoiceWorkflowBase
from pydantic import BaseModel

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.utils.utils import load_config

setup_logging()

logger = logging.getLogger(__name__)


class HealthStatusResult(BaseModel):
    """Result of the health status inquiry."""

    status: str
    transcription: str


class HealthStatusVoiceWorkflow(VoiceWorkflowBase):
    """Voice workflow for health status inquiry."""

    PROMPT = """
    You are a helpful and kind health status inquiry agent.
    <objective>
    Your job is to determine if a person is OK or needs help based on their response to the question "Is everything OK?"
    </objective>

    <rules>
    - Always be compassionate and considerate.
    - Focus only on analyzing the user's response for any signs of confusion, or calls for help.
    - If the user is confused or needs help, respond with "{health_status_needs_help}".
    - If the user is OK, respond with "{health_status_ok}".
    - If the user is unsure or the response cannot be determined, respond with "{health_status_not_ok}".
    </rules>
    """

    def __init__(self) -> None:
        """Initialize the health status voice workflow."""
        self.config = load_config()
        self.model = self.config["agent"]["health_status_inquiry"]["model"]
        self.health_status_ok = self.config["agent"]["person_state_analyzer"]["health_status_ok"]
        self.health_status_not_ok = self.config["agent"]["person_state_analyzer"]["health_status_not_ok"]
        self.health_status_needs_help = self.config["agent"]["person_state_analyzer"]["health_status_needs_help"]
        self.final_output: str = ""
        # Create a simple agent for analysis
        self.agent = Agent(
            name="Health Status Inquiry Agent",
            instructions=self.get_prompt(),
            model=self.model,
            output_type=HealthStatusResult,
        )

        # Initialize input history for the agent
        self._input_history = []
        self._current_agent = self.agent

    def get_prompt(self) -> str:
        return self.PROMPT.format(
            health_status_ok=self.health_status_ok,
            health_status_not_ok=self.health_status_not_ok,
            health_status_needs_help=self.health_status_needs_help,
        )

    async def run(self, transcription: str) -> AsyncIterator[str]:
        """
        Process the transcription from the user and determine health status.

        Args:
            transcription: The transcribed text from user's speech

        Yields:
            Appropriate responses to be converted to speech
        """
        logger.info(f"Processing transcription: {transcription}")

        self.final_output = self.health_status_needs_help

        # Add the transcription to input history
        self._input_history.append({"role": "system", "content": self.get_prompt()})
        self._input_history.append({"role": "user", "content": transcription})

        # Run the agent to analyze the health status - non-streamed
        from agents import Runner

        result = await Runner.run(self._current_agent, self._input_history)

        logger.info(f"Health status result: {result.final_output.status}")
        logger.info(f"Health status transcription: {result.final_output.transcription}")

        # Yield the status string instead of the HealthStatusResult object
        self.final_output = result.final_output.status
        yield result.final_output.status

        # Update input history for future interactions
        self._input_history = result.to_input_list()
        self._current_agent = result.last_agent
