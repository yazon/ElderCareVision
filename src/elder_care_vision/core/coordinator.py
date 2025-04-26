import asyncio
import base64
import logging
from dataclasses import dataclass, field
from enum import Enum, auto

from agents import Runner

from elder_care_vision.core.agents.health_status_inquiry import HealthStatusInquiryAgent
from elder_care_vision.core.agents.person_state_analyzer import PersonStateAnalyzerAgent
from elder_care_vision.core.tools.emergency_call.emergency_call import emergency_call_tool

# from elder_care_vision.core.tools.fall_camera_detector import fall_camera_detector_tool
from elder_care_vision.utils.utils import get_static_path, load_config

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
    image_data: dict | None = field(default=None)  # Data from fall detection
    psa_confidence: int | None = field(default=0)  # Confidence from person state analyzer
    last_psa_confidence: int | None = field(default=0)  # Last confidence from person state analyzer
    health_status: str | None = field(default=None)  # Health status from health status inquiry
    image_base64: str | None = field(default=None)  # Image base64 from fall detection
    # Add other shared data fields as needed


class Coordinator:
    """Class of Coordinator for an Elder Care Vision system."""

    def __init__(self) -> None:
        """Initializes the Coordinator."""
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
        self.person_state_analyzer_agent = PersonStateAnalyzerAgent()
        self.health_status_inquiry_agent = HealthStatusInquiryAgent()
        self.emergency_call_tool = emergency_call_tool

        # Use the absolute path to the static data directory
        static_path = get_static_path()
        img_path = static_path / "Elderly-Falls.jpg"
        img = img_path.read_bytes()
        self.context.image_base64 = base64.b64encode(img).decode("utf-8")

    @property
    def current_state(self) -> str:
        """Returns the name of the current state."""
        return self.context.current_state.name

    def transition_to_state(self, new_state: CoordinatorState) -> None:
        """Transitions to a new state."""
        # if new_state != self.context.current_state:
        self.context.current_state = new_state
        # logger.info(f"Transitioning to state: {new_state.name}")

    async def run(self) -> None:
        """Runs the coordinator state machine."""
        await self.person_state_analyzer_agent.run()
        while True:
            await self.process()

    async def process(self) -> None:
        """Processes the current state."""
        match self.context.current_state:
            # case CoordinatorState.START:
            #     await self.process_start_state()
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
        """Processes the analyzing image state."""
        if self.context.psa_confidence != self.context.last_psa_confidence:
            logger.info(f"PSA confidence: {self.context.psa_confidence}")
            self.context.last_psa_confidence = self.context.psa_confidence

            # Transition based on PSA confidence
            if self.context.psa_confidence > self.confidence_threshold_1:
                self.transition_to_state(CoordinatorState.CALLING_EMERGENCY)
            elif self.context.psa_confidence > self.confidence_threshold_2:
                self.transition_to_state(CoordinatorState.INQUIRING_HEALTH)
            else:
                self.transition_to_state(CoordinatorState.ANALYZING_IMAGE)

    async def process_inquiring_health_state(self) -> None:
        """Processes the inquiring health state."""
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
        """Processes the calling emergency state."""
        logger.info("---- BEGINNING OF CALLING EMERGENCY STATE ----")
        # TODO: make it in a thread
        await self.emergency_call_tool(self.context.image_base64)
        await asyncio.sleep(10)
        self.transition_to_state(CoordinatorState.ANALYZING_IMAGE)
