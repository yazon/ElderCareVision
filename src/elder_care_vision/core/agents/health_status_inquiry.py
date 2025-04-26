"""Placeholder for the Health Status Inquiry agent."""

import asyncio
import logging
import os

import numpy as np
import sounddevice as sd
from agents.voice import StreamedAudioInput, VoicePipeline, VoicePipelineConfig
from agents.voice.models.openai_stt import OpenAISTTModel, STTModelSettings
from agents.voice.models.openai_tts import OpenAITTSModel, TTSModelSettings
from openai import AsyncOpenAI

from elder_care_vision.core.agents.health_status_voice_workflow import HealthStatusVoiceWorkflow

logger = logging.getLogger(__name__)


class HealthStatusInquiryAgent:
    """Conducts inquiries about a person's health status."""

    # Audio configuration
    SAMPLE_RATE = 24000
    CHANNELS = 1
    FORMAT = np.int16
    RECORDING_TIMEOUT = 8  # Maximum recording time in seconds
    MAX_ITERATIONS = 2  # Maximum number of iterations

    def __init__(self) -> None:
        """Initializes the Health Status Inquiry agent using the BaseAgent."""
        logger.info("Initializing Health Status Inquiry Agent")
        self.iteration = 0
        self.workflow = HealthStatusVoiceWorkflow()
        self.audio_input = StreamedAudioInput()
        self.pipeline: VoicePipeline | None = None
        self.tts_settings: TTSModelSettings | None = None
        self.tts_model: OpenAITTSModel | None = None

    def _initialize_pipeline(self) -> None:
        """Initializes the voice pipeline, models, and configuration."""
        logger.info("Initializing voice pipeline...")

        # Set up OpenAI STT and TTS models
        openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        stt_model = OpenAISTTModel(model=self.workflow.stt_model, openai_client=openai_client)
        self.tts_model = OpenAITTSModel(model=self.workflow.tts_model, openai_client=openai_client)
        self.tts_settings = TTSModelSettings(**self.workflow.tts_settings)
        stt_settings = STTModelSettings(**self.workflow.stt_settings)

        # Create the voice pipeline
        self.pipeline = VoicePipeline(
            workflow=self.workflow,
            stt_model=stt_model,
            tts_model=self.tts_model,
            config=VoicePipelineConfig(tts_settings=self.tts_settings, stt_settings=stt_settings),
        )
        logger.info("Voice pipeline initialized.")

    async def _play_tts_prompt(self, output_stream: sd.OutputStream) -> None:
        """Generates and plays the TTS prompt for the current iteration."""
        if not self.workflow or not self.tts_model or not self.tts_settings:
            msg = "Workflow or TTS components not initialized."
            logger.error(msg)
            raise RuntimeError(msg)

        prompt = self.workflow.get_audio_prompt(self.iteration)
        logger.info(f"Playing TTS prompt: '{prompt}'")
        tts_audio_chunks = []
        async for chunk in self.tts_model.run(prompt, self.tts_settings):
            tts_audio_chunks.append(chunk)
        tts_audio = b"".join(tts_audio_chunks)

        # Play the audio
        output_stream.write(np.frombuffer(tts_audio, dtype=np.int16))
        output_stream.stop()
        logger.info(f"Played '{prompt}', now recording response")

    async def _record_and_process_response(
        self,
        input_stream: sd.InputStream,
    ) -> tuple[asyncio.Task | None, asyncio.Task | None]:
        """Starts audio recording and pipeline processing tasks."""
        if not self.pipeline or not self.audio_input:
            msg = "Pipeline or audio input not initialized."
            logger.error(msg)
            raise RuntimeError(msg)

        logger.info("Starting recording and pipeline processing...")
        input_stream.start()
        recording_task = asyncio.create_task(self.record_audio(input_stream, self.audio_input))
        pipeline_task = asyncio.create_task(self.pipeline.run(self.audio_input))
        return recording_task, pipeline_task

    async def _wait_for_tasks(self, recording_task: asyncio.Task, pipeline_task: asyncio.Task) -> None:
        """Waits for recording and pipeline tasks to complete, handling timeouts."""
        try:
            logger.info("Waiting for recording task...")
            await asyncio.wait_for(recording_task, timeout=self.RECORDING_TIMEOUT)
            logger.info("Recording task finished.")
        except TimeoutError:
            logger.warning("Recording timed out after maximum duration")
            if not recording_task.done():
                recording_task.cancel()
                try:
                    await recording_task
                except asyncio.CancelledError:
                    logger.info("Recording task cancelled successfully due to timeout.")
        finally:
            # Ensure pipeline task is awaited even if recording times out/errors
            if not pipeline_task.done():
                logger.info("Waiting for pipeline task...")
                try:
                    _ = await pipeline_task
                    logger.info("Pipeline task finished.")
                except Exception:
                    logger.exception("Error during pipeline processing after recording timeout/error.")
            else:
                # If it already finished (likely normal case or cancelled), check for exceptions
                try:
                    _ = pipeline_task.result()  # Re-raise exception if one occurred
                    logger.info("Pipeline task already finished.")
                except asyncio.CancelledError:
                    logger.info("Pipeline task was cancelled.")
                except Exception:
                    logger.exception("Error during pipeline processing.")

    def _cleanup_resources(
        self,
        output_stream: sd.OutputStream,
        input_stream: sd.InputStream,
        recording_task: asyncio.Task | None,
        pipeline_task: asyncio.Task | None,
    ) -> None:
        """Cleans up audio streams and cancels running tasks."""
        logger.info("Cleaning up resources for interaction...")
        # Cancel any remaining running tasks safely
        if recording_task and not recording_task.done():
            logger.info("Cancelling incomplete recording task during cleanup.")
            recording_task.cancel()
            # No await needed here, just ensuring it's cancelled

        if pipeline_task and not pipeline_task.done():
            logger.info("Cancelling incomplete pipeline task during cleanup.")
            pipeline_task.cancel()
            # No await needed here

        # Clean up audio streams
        try:
            if hasattr(output_stream, "active") and output_stream.active:
                output_stream.stop()
            output_stream.close()
            logger.debug("Output stream closed.")
        except Exception as e:
            logger.warning(f"Error closing output stream: {e}")

        try:
            if hasattr(input_stream, "active") and input_stream.active:
                input_stream.stop()
            input_stream.close()
            logger.debug("Input stream closed.")
        except Exception as e:
            logger.warning(f"Error closing input stream: {e}")
        logger.info("Resource cleanup complete.")

    async def _handle_interaction(self) -> None:
        """Handles a single interaction cycle: TTS -> Record -> STT -> Cleanup."""
        output_stream = sd.OutputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype=self.FORMAT)
        input_stream = sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype=self.FORMAT)
        recording_task = None
        pipeline_task = None

        try:
            # 1. Play TTS prompt
            output_stream.start()
            await self._play_tts_prompt(output_stream)  # output_stream stopped inside

            # 2. Start recording and processing
            recording_task, pipeline_task = await self._record_and_process_response(input_stream)

            # 3. Wait for tasks to complete (or timeout)
            if recording_task and pipeline_task:
                await self._wait_for_tasks(recording_task, pipeline_task)
            else:
                logger.error("Failed to create recording or pipeline tasks.")

        except Exception:
            logger.exception("Unhandled exception during voice interaction cycle.")
        finally:
            self._cleanup_resources(output_stream, input_stream, recording_task, pipeline_task)

    async def run_agent(self) -> str:
        """Runs the health status inquiry agent."""
        logger.info("Running health status inquiry agent...")
        self._initialize_pipeline()
        if not self.pipeline or not self.workflow or not self.tts_model or not self.tts_settings:
            msg = "Pipeline components not initialized correctly."
            logger.error(msg)
            raise RuntimeError(msg)

        while self.iteration < self.MAX_ITERATIONS:
            await self._handle_interaction()

            # Check workflow state after interaction
            if self.workflow.final_output not in (self.workflow.health_status_not_ok, ""):
                logger.info(f"User status determined as '{self.workflow.final_output}', breaking loop.")
                break
            logger.info("User status inconclusive or requires help, continuing loop if iterations remain.")

            self.iteration += 1

        # Determine final result after loop finishes or breaks
        if self.workflow.final_output == "":
            logger.info("Max iterations reached with no conclusive response. Assuming help is needed.")
            return self.workflow.health_status_needs_help

        return self.workflow.final_output

    async def record_audio(self, input_stream: sd.InputStream, audio_input: StreamedAudioInput) -> None:
        """
        Record audio from the microphone and send it to the audio input.

        Continues recording until either silence is detected for a period
        or the maximum recording time is reached.

        Args:
            input_stream: The sounddevice input stream
            audio_input: The StreamedAudioInput to send audio to
        """
        read_size = int(self.SAMPLE_RATE * 0.1)  # 100ms chunks
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

        except Exception:
            logger.exception("Error recording audio")
        finally:
            logger.info("Audio recording completed")


if __name__ == "__main__":
    agent = HealthStatusInquiryAgent()
    result = asyncio.run(agent.run_agent())
    logger.info(f"Health status inquiry result: {result}")
