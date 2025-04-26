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
    MAX_ITERATIONS = 2

    def __init__(self) -> None:
        """Initializes the Health Status Inquiry agent using the BaseAgent."""
        logger.info("Initializing Health Status Inquiry Agent")
        self.iteration = 0

    async def run_agent(self) -> str:
        """Runs the health status inquiry agent."""
        logger.info("Running health status inquiry agent...")
        # Create the workflow and audio input
        workflow = HealthStatusVoiceWorkflow()
        audio_input = StreamedAudioInput()

        # Initialize streams
        output_stream = sd.OutputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype=self.FORMAT)
        input_stream = sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype=self.FORMAT)

        # Set up OpenAI STT and TTS models
        openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        stt_model = OpenAISTTModel(model="gpt-4o-mini-transcribe", openai_client=openai_client)
        tts_model = OpenAITTSModel(model="gpt-4o-mini-tts", openai_client=openai_client)
        tts_settings = TTSModelSettings(voice="ash")
        stt_settings = STTModelSettings(language="en")

        # Create the voice pipeline
        pipeline = VoicePipeline(
            workflow=workflow,
            stt_model=stt_model,
            tts_model=tts_model,
            config=VoicePipelineConfig(tts_settings=tts_settings, stt_settings=stt_settings),
        )

        while self.iteration < self.MAX_ITERATIONS:
            output_stream = sd.OutputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype=self.FORMAT)
            input_stream = sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, dtype=self.FORMAT)
            recording_task = None
            pipeline_task = None
            try:
                # 1. Play initial TTS prompt "Is everything OK?"
                output_stream.start()

                # Use TTS model to generate the initial prompt
                prompt = workflow.get_audio_prompt(self.iteration)
                logger.info(f"Playing TTS prompt: '{prompt}'")
                tts_audio_chunks = []
                async for chunk in tts_model.run(prompt, tts_settings):
                    tts_audio_chunks.append(chunk)
                tts_audio = b"".join(tts_audio_chunks)

                # Play the audio
                output_stream.write(np.frombuffer(tts_audio, dtype=np.int16))
                output_stream.stop()

                logger.info(f"Played '{prompt}', now recording response")

                # 2. Start recording microphone input
                input_stream.start()
                recording_task = asyncio.create_task(self.record_audio(input_stream, audio_input))

                # 3. Process the audio through the pipeline
                pipeline_task = asyncio.create_task(pipeline.run(audio_input))

                # Wait for the recording to complete
                try:
                    await asyncio.wait_for(recording_task, timeout=self.RECORDING_TIMEOUT)
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
                _ = await pipeline_task

            except Exception:
                logger.exception("Error in health inquiry voice interaction")

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

            # Only break if user needs immediate help or is fine
            if workflow.final_output != workflow.health_status_not_ok:
                logger.info("User is fine, breaking!")
                break

            self.iteration += 1

        return workflow.final_output

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
