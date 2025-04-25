import os

from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem


class OpenAIChatCompletionValidationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class OpenAIChatCompletionError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class OpenAIImageGenerationError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class OpenAIService:
    def __init__(self, max_tokens: int = 4096) -> None:
        """
        Initializes the OpenAIService.

        Args:
            max_tokens: The maximum number of tokens for responses.
        """
        self.max_tokens = max_tokens
        self.openai = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def get_client(self) -> AsyncOpenAI:
        """Returns the underlying AsyncOpenAI client."""
        return self.openai

    async def chat(
        self,
        messages: list[dict],
        model: str = "gpt-4.1",
        tool_choice: str = "none",
        stream: bool = False,
        temperature: float = -1,
        json_mode: bool = False,
        reasoning: dict | None = None,
        tools: list | None = None,
        instructions: str | None = None,
    ) -> dict:
        """
        Send a chat completion request to OpenAI.

        Args:
            messages: List of message dictionaries to send to the API
            model: The OpenAI model to use
            reasoning: Optional configuration for o-series models reasoning capabilities.
                       Can include 'effort' (low/medium/high) and 'generate_summary' (concise/detailed).
            tools: List of tools to make available to the model
            tool_choice: Controls tool usage. Can be "none", "auto", or "required"
            stream: Whether to stream the response
            temperature: Temperature for response generation, -1 means use default
            json_mode: Whether to request JSON output
            instructions: Optional system-level instructions for the model. Provided as the first message.

        Returns:
            The response from the OpenAI API as a dictionary.
        """
        # Validate that each message is a dictionary with "role" and "content"
        for message in messages:
            if not isinstance(message, dict) or "role" not in message or "content" not in message:
                msg = "Each message must be a dictionary with 'role' and 'content' keys."
                raise OpenAIChatCompletionValidationError(msg)

        return await self.__response(
            messages=messages,
            model=model,
            reasoning=reasoning,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            temperature=temperature,
            json_mode=json_mode,
            instructions=instructions,
        )

    async def __response(
        self,
        messages: list[dict],
        model: str,
        reasoning: dict | None = None,
        store: bool = True,
        tools: list | None = None,
        tool_choice: str = "none",
        stream: bool = False,
        temperature: float = -1,
        json_mode: bool = False,
        instructions: str | None = None,
    ) -> dict:
        """
        Send a request to the OpenAI API and handle the response.

        Args:
            messages: List of message dictionaries to send to the API
            model: The OpenAI model to use
            reasoning: Optional configuration for o-series models reasoning capabilities.
                       Can include 'effort' (low/medium/high) and 'generate_summary' (concise/detailed).
            store: Whether to store the generated model response for later retrieval via API.
            tools: List of tools to make available to the model
            tool_choice: Controls tool usage. Can be "none", "auto", or "required"
            stream: Whether to stream the response
            temperature: Temperature for response generation, -1 means use default
            json_mode: Whether to request JSON output
            instructions: Optional system-level instructions for the model. Provided as the first message.

        Returns:
            The response from the OpenAI API as a dictionary

        Raises:
            OpenAIChatCompletionValidationError: If validation fails
            OpenAIChatCompletionError: If the API call fails
        """
        try:
            self.__validate_reasoning(reasoning)  # Added reasoning validation call
            self.__validate_tool_choice(tool_choice)

            # Make API call
            api_response = await self.__make_openai_api_call(
                messages, model, reasoning, store, tools, tool_choice, stream, temperature, json_mode, instructions
            )

            # Extract content from response
            content, reasoning_content, message = self.__extract_response_content(api_response, reasoning)

            # Build structured response and return it
            return self.__build_response_object(content, message, model, api_response, reasoning, reasoning_content)

        except OpenAIError as error:
            return self.__handle_openai_error(error)

    def __validate_reasoning(self, reasoning: dict | None = None) -> None:
        """
        Validate the reasoning configuration if provided.

        Args:
            reasoning: Optional reasoning configuration

        Raises:
            OpenAIChatCompletionValidationError: If reasoning configuration is invalid
        """
        if reasoning is not None:
            if "effort" in reasoning and reasoning["effort"] not in ["low", "medium", "high"]:
                msg = f"Invalid reasoning effort: {reasoning["effort"]}. Must be one of ['low', 'medium', 'high']"
                raise OpenAIChatCompletionValidationError(msg)
            if "generate_summary" in reasoning and reasoning["generate_summary"] not in ["concise", "detailed"]:
                msg = f"Invalid reasoning summary type: {reasoning["generate_summary"]}. Must be one of ['concise', 'detailed']"
                raise OpenAIChatCompletionValidationError(msg)

    def __validate_tool_choice(self, tool_choice: str) -> None:
        """
        Validate the tool_choice parameter.

        Args:
            tool_choice: The tool choice parameter to validate

        Raises:
            OpenAIChatCompletionValidationError: If tool_choice is invalid
        """
        valid_tool_choices = ["none", "auto", "required"]
        if tool_choice not in valid_tool_choices:
            msg = f"Invalid tool_choice: {tool_choice}. Must be one of {valid_tool_choices}"
            raise OpenAIChatCompletionValidationError(msg)

    async def __make_openai_api_call(
        self,
        messages: list[dict],
        model: str,
        reasoning: dict | None,
        store: bool,
        tools: list | None,
        tool_choice: str,
        stream: bool,
        temperature: float,
        json_mode: bool,
        instructions: str | None = None,
    ) -> any:
        """
        Make API call to OpenAI.

        Args:
            messages: List of message dictionaries to send to the API
            model: The OpenAI model to use
            reasoning: Optional reasoning configuration
            store: Whether to store the generated model response
            tools: List of tools to make available to the model
            tool_choice: Controls tool usage
            stream: Whether to stream the response
            temperature: Temperature for response generation
            json_mode: Whether to request JSON output
            instructions: Optional system-level instructions for the model. Provided as the first message.

        Returns:
            OpenAI API response
        """
        return await self.openai.responses.create(
            input=messages,
            model=model,
            reasoning=reasoning,
            store=store,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream,
            temperature=temperature if temperature >= 0 else None,
            text={"format": {"type": "json_object" if json_mode else "text"}},
            instructions=instructions,
        )

    def __extract_response_content(self, api_response: any, reasoning: dict | None = None) -> tuple[str, str, any]:
        """
        Extract content from the OpenAI API response.

        Args:
            api_response: The response from the OpenAI API
            reasoning: Optional reasoning configuration

        Returns:
            Tuple containing (content, reasoning_content, message)
        """
        content = ""
        reasoning_content = ""
        message = {"role": "assistant"}

        if api_response.output and len(api_response.output) > 0:
            # When reasoning is enabled, the response can contain both reasoning information and message content
            # Check each output item to determine its type and extract appropriate content
            for output_item in api_response.output:
                # Check if this is a standard message
                if isinstance(output_item, ResponseOutputMessage):
                    message = output_item
                    # print(message)
                    for content_item in output_item.content:
                        if content_item.type == "output_text":
                            content += content_item.text
                # Check if this is a reasoning item
                elif isinstance(output_item, ResponseReasoningItem):
                    reasoning_item = output_item
                    # print(reasoning_item)
                    # Extract text from reasoning summaries if available
                    if hasattr(reasoning_item, "summary") and reasoning_item.summary:
                        reasoning_content = " ".join(summary.text for summary in reasoning_item.summary)
                    else:
                        # Fallback to string representation
                        reasoning_content = str(reasoning_item)

            # If we didn't find a standard message but have reasoning data, use fallback handling
            if (
                not content and reasoning is not None and hasattr(api_response, "reasoning") and api_response.reasoning
            ):  # Check api_response.reasoning directly
                # For ResponseReasoningItem objects check if it has summaries
                if isinstance(api_response.reasoning, ResponseReasoningItem) and api_response.reasoning.summary:
                    content = " ".join(summary.text for summary in api_response.reasoning.summary)
                elif isinstance(api_response.reasoning, dict) and "text" in api_response.reasoning:
                    content = api_response.reasoning["text"]
                else:
                    # Fallback to string representation
                    content = str(api_response.reasoning)

            # If we didn't find any message objects, try to find the first ResponseOutputMessage
            if "role" not in message or message["role"] != "assistant":
                found_message = False
                for output_item in api_response.output:
                    if isinstance(output_item, ResponseOutputMessage):
                        message = output_item
                        found_message = True
                        break
                # If still no message, use a default structure or handle as error
                if not found_message:
                    # Depending on requirements, might need to raise an error or provide default
                    print("Warning: No ResponseOutputMessage found in API response output.")
                    message = {"role": "assistant"}  # Fallback to default

        return content, reasoning_content, message

    def __build_response_object(
        self, content: str, message: any, model: str, api_response: any, reasoning: dict | None, reasoning_content: str
    ) -> dict:
        """
        Build a structured response object from the API response.

        Args:
            content: Extracted content from the response
            message: The message object from the response
            model: The OpenAI model used
            api_response: The original API response
            reasoning: Optional reasoning configuration
            reasoning_content: Extracted reasoning content

        Returns:
            Structured response dictionary
        """
        # Create a structured response that maintains compatibility with existing code
        response = {
            "content": content,
            "role": message.role if hasattr(message, "role") else "assistant",
            "model": model,
            "id": api_response.id,
            "created_at": api_response.created_at,
            "status": api_response.status,
            "usage": {
                "prompt_tokens": api_response.usage.input_tokens,
                "completion_tokens": api_response.usage.output_tokens,
                "total_tokens": api_response.usage.total_tokens,
            },
        }

        # Include reasoning tokens information for o-series models
        if hasattr(api_response.usage, "completion_tokens_details") and hasattr(
            api_response.usage.completion_tokens_details, "reasoning_tokens"
        ):
            response["usage"]["reasoning_tokens"] = api_response.usage.completion_tokens_details.reasoning_tokens

        # Include reasoning information if available
        if reasoning is not None:
            if api_response.reasoning:
                response["reasoning"] = api_response.reasoning
            if reasoning_content:
                response["reasoning_text"] = reasoning_content

        # Check for incomplete status due to token limits
        if (
            api_response.status == "incomplete"
            and hasattr(api_response, "incomplete_details")
            and api_response.incomplete_details.reason == "max_output_tokens"
        ):
            print("Ran out of tokens")
            if response["content"]:
                print("Partial output:", response["content"])
            else:
                print("Ran out of tokens during reasoning")

        return response

    def __handle_openai_error(self, error: OpenAIError) -> None:
        """
        Handle OpenAI API errors.

        Args:
            error: The OpenAI error that occurred

        Raises:
            OpenAIChatCompletionError: Rethrows the error with additional context
        """
        print("Error in OpenAI completion:", error)
        raise OpenAIChatCompletionError(str(error)) from error

    async def text_to_speech(self, text: str, voice: str = "alloy", output_format: str = "mp3") -> bytes:
        """Convert text to speech using OpenAI's TTS API.

        Args:
            text: The text to convert to speech
            voice: The voice to use (alloy, echo, fable, onyx, nova, shimmer)
            output_format: The output format (mp3, opus, aac, flac)

        Returns:
            bytes: The audio data in the specified format

        Raises:
            OpenAIChatCompletionError: If the text-to-speech conversion fails
        """
        try:
            response = await self.openai.audio.speech.create(
                model="tts-1", voice=voice, input=text, response_format=output_format
            )
            return response.content
        except Exception as e:
            raise OpenAIChatCompletionError(f"Failed to convert text to speech: {str(e)}") from e
