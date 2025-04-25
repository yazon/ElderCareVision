"""Base class for agents."""

import logging
from abc import ABC

from agents import Agent
from pydantic import BaseModel

from elder_care_vision.utils.utils import load_config

logger = logging.getLogger(__name__)


class BaseAgent(ABC):  # noqa: B024
    """Abstract base class for specialized agents."""

    PROMPT: str = "You are a helpful assistant."  # Default prompt, should be overridden

    def __init__(self, agent_name: str, output_type: BaseModel) -> None:
        """
        Initializes the Base Agent.

        Args:
            agent_name: The specific name of the agent (e.g., "person_state_analyzer").
                        Used to load configuration.
            output_type: The output type of the agent.
        """
        logger.info(f"Initializing Base Agent for: {agent_name}")
        self.agent_name = agent_name
        self.config = load_config()
        try:
            agent_config = self.config["agent"][self.agent_name]
            self.model = agent_config["model"]
            self.temperature = agent_config["temperature"]
        except KeyError as e:
            logger.exception(f"Missing configuration for agent '{self.agent_name}'")
            msg = f"Configuration error for agent '{self.agent_name}'"
            raise ValueError(msg) from e

        # Note: self.get_prompt() is called here, which uses the PROMPT
        # class variable defined in the *subclass*.
        self.agent = Agent(
            name=self.agent_name,
            model=self.model,
            instructions=self.get_prompt(),
            output_type=output_type,
        )
        logger.info(f"Agent '{self.agent_name}' initialized successfully.")

    def get_agent(self) -> Agent:
        """Returns the underlying agent instance."""
        return self.agent

    def get_prompt(self) -> str:
        """
        Returns the specific prompt for the agent.

        This method relies on the `PROMPT` class variable being defined
        in the inheriting subclass.
        """
        if not hasattr(self.__class__, "PROMPT") or self.PROMPT == BaseAgent.PROMPT:
            logger.warning(
                f"Agent '{self.agent_name}' is using the default BaseAgent prompt. "
                "Define a PROMPT class variable in the subclass."
            )
        # Access PROMPT via the class to ensure the subclass's version is used
        return self.__class__.PROMPT
