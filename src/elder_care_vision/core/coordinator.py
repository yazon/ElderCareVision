import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

from elder_care_vision.core.agents.health_status_inquiry import HealthStatusInquiryAgent
from elder_care_vision.core.agents.person_state_analyzer import PersonStateAnalyzerAgent
from elder_care_vision.core.agents.psa_data import FallDetectionResult
from elder_care_vision.core.tools.emergency_call.emergency_call import emergency_call_tool

# from elder_care_vision.core.tools.fall_camera_detector import fall_camera_detector_tool
from elder_care_vision.utils.utils import load_config

logger = logging.getLogger(__name__)


class CoordinatorState(Enum):
    """Defines the possible states of the Elder Care Vision system."""

    # START = auto()
    ANALYZING_IMAGE = auto()
    INQUIRING_HEALTH = auto()
    CALLING_EMERGENCY = auto()


@dataclass
class CoordinatorContext:
    """Holds the context for the Coordinator, including state and shared data."""

    current_state: CoordinatorState = CoordinatorState.ANALYZING_IMAGE
    last_psa_confidence: int | None = field(default=0)  # Last confidence from person state analyzer
    health_status: str | None = field(default=None)  # Health status from health status inquiry
    fall_detection_result: FallDetectionResult | None = field(
        default=None
    )  # Fall detection result from person state analyzer
    # Add other shared data fields as needed


class Coordinator:
    """
    Class of Coordinator for an Elder Care Vision system.

    Manages the state transitions and coordinates the actions of different agents
    (PersonStateAnalyzerAgent, HealthStatusInquiryAgent) and tools (emergency_call_tool)
    based on the detected situation and health status inquiries.

    Attributes:
        config (dict): Configuration loaded from the system's config file.
        confidence_threshold_1 (int): High confidence threshold for fall detection.
        confidence_threshold_2 (int): Medium confidence threshold for fall detection.
        emergency_statuses (tuple[str, ...]): Tuple of health statuses considered emergencies.
        context (CoordinatorContext): Holds the current state and shared data.
        person_state_analyzer_agent (PersonStateAnalyzerAgent): Agent for analyzing person state.
        health_status_inquiry_agent (HealthStatusInquiryAgent): Agent for inquiring health status.
        emergency_call_tool (callable): Tool function to make emergency calls.
    """

    def __init__(self, video_source: int | str = 0) -> None:
        """
        Initializes the Coordinator.

        Loads configuration, sets thresholds, initializes context, and instantiates agents/tools.

        Args:
            video_source: Either a camera ID (integer) or an RTSP stream URL (string).
                         Defaults to 0 (default camera).
        """
        logger.info("Initializing Coordinator State Machine")
        self.config = load_config()
        self.confidence_threshold_1 = self.config["agent"]["person_state_analyzer"]["confidence_threshold_1"]
        self.confidence_threshold_2 = self.config["agent"]["person_state_analyzer"]["confidence_threshold_2"]
        self.emergency_statuses = (
            self.config["agent"]["health_status_inquiry"]["health_status_needs_help"],
            self.config["agent"]["health_status_inquiry"]["health_status_not_ok"],
        )
        self.context = CoordinatorContext()  # Initialize context
        print(f"Initializing state machine in state: {self.context.current_state.name}")
        self.person_state_analyzer_agent = PersonStateAnalyzerAgent(video_source)
        self.health_status_inquiry_agent = HealthStatusInquiryAgent()
        self.emergency_call_tool = emergency_call_tool

    @property
    def current_state(self) -> str:
        """
        Returns the name of the current state.

        Returns:
            str: The name of the current CoordinatorState enum member.
        """
        return self.context.current_state.name

    def transition_to_state(self, new_state: CoordinatorState) -> None:
        """
        Transitions the coordinator to a new state.

        Logs the transition and resets specific agent properties if necessary
        (e.g., resetting confidence level when returning to ANALYZING_IMAGE).

        Args:
            new_state (CoordinatorState): The target state to transition to.
        """
        if new_state != self.context.current_state:
            self.context.current_state = new_state
            logger.info(f"Transitioning to state: {new_state.name}")
            if new_state == CoordinatorState.ANALYZING_IMAGE:
                # Reinit confidence level
                self.person_state_analyzer_agent.fall_detection_result.confidence_level = 0

    async def run(self) -> None:
        """
        Runs the main loop of the coordinator state machine.

        Initializes the person state analyzer and continuously processes the current state.
        This method runs indefinitely.
        """
        await self.person_state_analyzer_agent.run()
        while True:
            await self.process()

    async def process(self) -> None:
        """
        Processes the logic associated with the current state.

        Calls the appropriate state processing method based on the current context state.
        Handles unknown states by logging an error and transitioning to a default state.
        """
        match self.context.current_state:
            case CoordinatorState.ANALYZING_IMAGE:
                await self.process_analyzing_image_state()
            case CoordinatorState.INQUIRING_HEALTH:
                await self.process_inquiring_health_state()
            case CoordinatorState.CALLING_EMERGENCY:
                await self.process_calling_emergency_state()
            case _:
                logger.error(f"FATAL: Unknown state: {self.context.current_state}")
                self.transition_to_state(CoordinatorState.START)

    async def process_analyzing_image_state(self) -> None:
        """
        Processes the ANALYZING_IMAGE state.

        Retrieves the latest confidence level from the PersonStateAnalyzerAgent.
        If the confidence level changes, updates the context and determines the next state
        based on configured confidence thresholds.
        Transitions to CALLING_EMERGENCY for high confidence, INQUIRING_HEALTH for medium confidence,
        or stays in ANALYZING_IMAGE for low confidence.
        """
        confidence_level = self.person_state_analyzer_agent.fall_detection_result.confidence_level
        self.person_state_analyzer_agent.fall_detection_result.confidence_level = (
            -1
        )  # set to negative value to indicate that the value was read
        logger.info(f"PSA confidence: {confidence_level} (previous: {self.context.last_psa_confidence})")
        await asyncio.sleep(0.5)
        if confidence_level != self.context.last_psa_confidence:
            logger.info(f"Confidence level changed from {self.context.last_psa_confidence} to {confidence_level}")
            self.context.last_psa_confidence = confidence_level
            # Store the fall detection result to context for emergency call
            self.context.fall_detection_result = self.person_state_analyzer_agent.fall_detection_result

            # Transition based on PSA confidence
            if confidence_level > self.confidence_threshold_1:
                logger.info(
                    f"High confidence ({confidence_level} > {self.confidence_threshold_1}), transitioning to CALLING_EMERGENCY"
                )
                self.transition_to_state(CoordinatorState.CALLING_EMERGENCY)
            elif confidence_level >= self.confidence_threshold_2:
                logger.info(
                    f"Medium confidence ({confidence_level} >= {self.confidence_threshold_2}), transitioning to INQUIRING_HEALTH"
                )
                self.transition_to_state(CoordinatorState.INQUIRING_HEALTH)
            else:
                logger.info(
                    f"Low confidence ({confidence_level} < {self.confidence_threshold_2}), staying in ANALYZING_IMAGE"
                )
                self.transition_to_state(CoordinatorState.ANALYZING_IMAGE)

    async def process_inquiring_health_state(self) -> None:
        """
        Processes the INQUIRING_HEALTH state.

        Runs the HealthStatusInquiryAgent to get the person's health status.
        Updates the context with the obtained health status.
        Transitions to CALLING_EMERGENCY if the status indicates an emergency,
        otherwise transitions back to ANALYZING_IMAGE.
        """
        logger.info("---- BEGINNING OF INQUIRING HEALTH STATE ----")
        # Use the health_status_inquiry_agent and context data
        hsa = HealthStatusInquiryAgent()
        self.context.health_status = await hsa.run_agent()
        logger.info(f"Health status: {self.context.health_status}")
        if self.context.health_status in self.emergency_statuses:
            self.transition_to_state(CoordinatorState.CALLING_EMERGENCY)
        else:
            self.transition_to_state(CoordinatorState.ANALYZING_IMAGE)

    async def process_calling_emergency_state(self) -> None:
        """
        Processes the CALLING_EMERGENCY state.

        Initiates an emergency call using the emergency_call_tool, providing
        the fall detection results and health status from the context.
        After the call attempt, transitions back to the ANALYZING_IMAGE state.
        """
        logger.info("---- BEGINNING OF CALLING EMERGENCY STATE ----")
        await self.emergency_call_tool(self.context.fall_detection_result, self.context.health_status)
        self.transition_to_state(CoordinatorState.ANALYZING_IMAGE)
