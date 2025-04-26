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
    """
    Result model for health status inquiry containing analysis results.

    Attributes:
        status: Categorized health status (OK/NeedsHelp/NotOK)
        transcription: Cleaned version of user's response transcription
    """

    status: str
    transcription: str


class HealthStatusVoiceWorkflow(VoiceWorkflowBase):
    """
    Voice workflow for health status inquiry using AI agent analysis.

    Handles the complete voice interaction flow including:
    - Speech-to-text conversion of user responses
    - Health status analysis using LLM agent
    - Text-to-speech response generation
    - Conversation history management
    """

    PROMPT = """
    You are a helpful and kind health status inquiry agent.
    <objective>
    Your job is to determine if a person is OK or needs help based on their response to the question "Is everything OK?"
    </objective>

    <rules>
    - Always be compassionate and considerate.
    - Always assume that user should respond in English.
    - Focus only on analyzing the user's response for any signs of confusion, or calls for help.
    - If the user needs help, respond with "{health_status_needs_help}".
    - If the user is OK, respond with "{health_status_ok}".
    - If the user is response cannot be determined or is not clear, respond with "{health_status_not_ok}".
    </rules>

    <examples>
    Example 1:
    - Assistant: "Is everything OK?"
    - User: "I'm not feeling well."
    - Assistant: "{health_status_needs_help}" (because the user is not feeling well)

    Example 2:
    - Assistant: "Is everything OK?"
    - User: "Yeah, I'm fine."
    - Assistant: "{health_status_ok}" (because the user is fine)

    Example 3:
    - Assistant: "Is everything OK?"
    - User: "I'm not sure."
    - Assistant: "{health_status_not_ok}" (because the user is not sure)

    Example 4:
    - Assistant: "Is everything OK?"
    - User: "جیسے میں کہوں."
    - Assistant: "{health_status_not_ok}" (because transcription is not in English)

    Example 5:
    - Assistant: "Is everything OK?"
    - User: "Nie wiem, por favor!"
    - Assistant: "{health_status_not_ok}" (because transcription is not in English)

    Example 6:
    - Assistant: "Is everything OK?"
    - User: ""
    - Assistant: "{health_status_not_ok}" (because there is no transcription - silence)
    </examples>
    """

    def __init__(self) -> None:
        """Initialize the health status voice workflow."""
        self.config = load_config()
        self.model = self.config["agent"]["health_status_inquiry"]["model"]
        self.health_status_ok = self.config["agent"]["health_status_inquiry"]["health_status_ok"]
        self.health_status_not_ok = self.config["agent"]["health_status_inquiry"]["health_status_not_ok"]
        self.health_status_needs_help = self.config["agent"]["health_status_inquiry"]["health_status_needs_help"]
        self.initial_ask_prompt = self.config["agent"]["health_status_inquiry"]["initial_ask_prompt"]
        self.retry_ask_prompt = self.config["agent"]["health_status_inquiry"]["retry_ask_prompt"]
        self.final_output: str = ""
        self.tts_model = self.config["agent"]["health_status_inquiry"]["tts_model"]
        self.stt_model = self.config["agent"]["health_status_inquiry"]["stt_model"]
        self.tts_settings = self.config["agent"]["health_status_inquiry"]["tts_settings"]
        self.stt_settings = self.config["agent"]["health_status_inquiry"]["stt_settings"]
        # Create agent for user message analysis
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
        """
        Format and return the system prompt with current configuration values.

        Returns:
            str: Fully formatted system prompt with injected config values
        """
        return self.PROMPT.format(
            health_status_ok=self.health_status_ok,
            health_status_not_ok=self.health_status_not_ok,
            health_status_needs_help=self.health_status_needs_help,
        )

    def get_audio_prompt(self, iteration: int) -> str:
        """
        Get appropriate voice prompt based on conversation iteration.

        Args:
            iteration: Current interaction attempt count (0=first try)

        Returns:
            str: Initial ask prompt for iteration 0, retry prompt otherwise
        """
        if iteration > 0:
            return self.retry_ask_prompt
        return self.initial_ask_prompt

    async def run(self, transcription: str) -> AsyncIterator[str]:
        """
        Process user response and generate appropriate health status assessment.

        Args:
            transcription: Raw text input from user's voice response

        Yields:
            str: Categorized health status string from analysis

        Notes:
            Updates conversation history and agent state after each interaction
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
