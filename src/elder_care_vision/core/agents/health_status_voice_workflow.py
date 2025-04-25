"""Voice workflow for the Health Status Inquiry agent using OpenAI agents SDK."""

import asyncio
import logging
import os
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd
from agents import Agent
from agents.voice import StreamedAudioInput, VoicePipeline, VoicePipelineConfig, VoiceWorkflowBase
from agents.voice.models.openai_stt import OpenAISTTModel, STTModelSettings
from agents.voice.models.openai_tts import OpenAITTSModel, TTSModelSettings
from openai import AsyncOpenAI
from pydantic import BaseModel

from elder_care_vision.config.logging_config import setup_logging
from elder_care_vision.utils.utils import load_config

setup_logging()

logger = logging.getLogger(__name__)


# Audio configuration
SAMPLE_RATE = 24000
CHANNELS = 1
FORMAT = np.int16
RECORDING_TIMEOUT = 8  # Maximum recording time in seconds


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
    - If the user is confused or needs help, respond with "{{self.health_status_needs_help}}".
    - If the user is OK, respond with "{{self.health_status_ok}}".
    - If the user is unsure or the response cannot be determined, respond with "{{self.health_status_not_ok}}".
    </rules>
    """

    def __init__(self):
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


async def run_health_inquiry_voice_interaction() -> str:
    """
    Run the complete health status inquiry voice interaction:
    1. Ask if everything is OK via TTS
    2. Record and transcribe the response
    3. Analyze the transcription and return the result
    """
    logger.info("Starting health status voice interaction")

    # Create the workflow and audio input
    workflow = HealthStatusVoiceWorkflow()
    audio_input = StreamedAudioInput()

    # Initialize streams
    output_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=FORMAT)

    input_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=FORMAT)

    # Set up OpenAI STT and TTS models
    openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    stt_model = OpenAISTTModel(model="gpt-4o-mini-transcribe", openai_client=openai_client)
    tts_model = OpenAITTSModel(model="gpt-4o-mini-tts", openai_client=openai_client)

    # Create the voice pipeline
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=stt_model,
        tts_model=tts_model,
        config=VoicePipelineConfig(
            tts_settings=TTSModelSettings(voice="ash"), stt_settings=STTModelSettings(language="en")
        ),
    )

    transcription = ""
    status = "inconclusive"
    recording_task = None
    pipeline_task = None

    try:
        # 1. Play initial TTS prompt "Is everything OK?"
        output_stream.start()

        # Use TTS model to generate the initial prompt
        initial_prompt = "   Hey! Is everything OK?"
        logger.info(f"Playing TTS prompt: '{initial_prompt}'")
        tts_audio_chunks = []
        async for chunk in tts_model.run(initial_prompt, TTSModelSettings(voice="alloy")):
            tts_audio_chunks.append(chunk)
        tts_audio = b"".join(tts_audio_chunks)

        # Play the audio
        output_stream.write(np.frombuffer(tts_audio, dtype=np.int16))
        output_stream.stop()

        logger.info("Played initial prompt, now recording response")

        # 2. Start recording microphone input
        input_stream.start()
        recording_task = asyncio.create_task(record_audio(input_stream, audio_input))

        # 3. Process the audio through the pipeline
        pipeline_task = asyncio.create_task(pipeline.run(audio_input))

        # Wait for the recording to complete
        try:
            await asyncio.wait_for(recording_task, timeout=RECORDING_TIMEOUT)
        except TimeoutError:
            logger.warning("Recording timed out after maximum duration")
            if recording_task and not recording_task.done():
                recording_task.cancel()
                try:
                    await recording_task
                except asyncio.CancelledError:
                    logger.info("Recording task cancelled successfully")

        # Process the pipeline result
        logger.info("Recording complete. Processing through STT...")
        pipeline_result = await pipeline_task

    except Exception as e:
        logger.exception(f"Error in health inquiry voice interaction: {e}")
        status = "inconclusive"

    finally:
        # Cancel any running tasks
        if recording_task and not recording_task.done():
            recording_task.cancel()

        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()

        # Clean up resources
        if hasattr(output_stream, "active") and output_stream.active:
            output_stream.stop()
        output_stream.close()

        if hasattr(input_stream, "active") and input_stream.active:
            input_stream.stop()
        input_stream.close()

    return workflow.final_output


async def record_audio(input_stream, audio_input: StreamedAudioInput) -> None:
    """
    Record audio from the microphone and send it to the audio input.
    Continues recording until either silence is detected for a period
    or the maximum recording time is reached.

    Args:
        input_stream: The sounddevice input stream
        audio_input: The StreamedAudioInput to send audio to
    """
    read_size = int(SAMPLE_RATE * 0.1)  # 100ms chunks
    silence_threshold = 2.0
    silence_duration = 0
    max_silence_duration = 1.0  # Stop after 1 seconds of silence

    logger.info("Started audio recording...")
    is_recording = True
    speech_detected = False

    try:
        while is_recording:
            # Read audio chunk
            data, overflowed = input_stream.read(read_size)

            if overflowed:
                logger.warning("Input overflowed")

            # Check if this is silence
            audio_level = np.abs(data).mean()
            is_silence = audio_level < silence_threshold

            # For logging
            if audio_level > 0.1 and not speech_detected:
                speech_detected = True
                logger.info(f"Speech detected (level: {audio_level:.4f})")

            # Send the audio data to the STT pipeline
            await audio_input.add_audio(data)

            # Update silence tracking
            if is_silence:
                silence_duration += 0.1  # 100ms chunk
                if silence_duration >= max_silence_duration and speech_detected:
                    logger.info("Detected end of speech (silence)")
                    is_recording = False
            else:
                silence_duration = 0

            # Small sleep to prevent busy waiting
            await asyncio.sleep(0.01)

    except Exception as e:
        logger.exception(f"Error recording audio: {e}")
    finally:
        logger.info("Audio recording completed")


# For standalone testing
async def main():
    """Run the health status inquiry as a standalone test."""
    print("\n=== STARTING HEALTH STATUS VOICE INTERACTION ===")
    print("The system will ask 'Is everything OK?' then record your response.")
    print("Please speak clearly after the prompt.\n")

    result = await run_health_inquiry_voice_interaction()

    print("\n=== RESULTS ===")
    print(f"Health Status: {result}")
    print("================")

    return result


if __name__ == "__main__":
    asyncio.run(main())
