#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2025 g-eoj
import abc
import json
import logging
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
import json_repair
import mlx.core as mx
import mlx.nn as nn
import openai
import tenacity

from .tools import Tool
from .utils import _is_package_available, encode_image_base64, is_url, make_image_url


if TYPE_CHECKING:
    import mlx_lm
    import mlx_vlm


logger = logging.getLogger(__name__)


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items() if k != ignore_key}
        return obj

    return convert(obj)


@dataclass
class ChatMessageToolCallDefinition:
    arguments: Any
    name: str
    description: Optional[str] = None

    @classmethod
    def from_hf_api(cls, tool_call_definition) -> "ChatMessageToolCallDefinition":
        return cls(
            arguments=tool_call_definition.arguments,
            name=tool_call_definition.name,
            description=tool_call_definition.description,
        )


@dataclass
class ChatMessageToolCall:
    function: ChatMessageToolCallDefinition
    id: str
    type: str

    @classmethod
    def from_hf_api(cls, tool_call) -> "ChatMessageToolCall":
        return cls(
            function=ChatMessageToolCallDefinition.from_hf_api(tool_call.function),
            id=tool_call.id,
            type=tool_call.type,
        )


@dataclass
class ChatMessage:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[ChatMessageToolCall]] = None
    raw: Optional[Any] = None  # Stores the raw output from the API
    finish_reason: Optional[str] = None

    def model_dump_json(self):
        return json.dumps(get_dict_from_nested_dataclasses(self, ignore_key="raw"))

    @classmethod
    def from_hf_api(cls, message, raw) -> "ChatMessage":
        tool_calls = None
        if getattr(message, "tool_calls", None) is not None:
            tool_calls = [ChatMessageToolCall.from_hf_api(tool_call) for tool_call in message.tool_calls]
        return cls(role=message.role, content=message.content, tool_calls=tool_calls, raw=raw)

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallDefinition(**tc["function"]), id=tc["id"], type=tc["type"]
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(**data)

    def dict(self):
        return json.dumps(get_dict_from_nested_dataclasses(self))


@dataclass
class MLXVLMGenerationResult:
    text: str
    token: Optional[int]
    logprobs: Optional[List[float]]
    prompt_tokens: int
    generation_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory: float
    finish_reason: Optional[str] = None


def parse_json_if_needed(arguments: Union[str, dict]) -> Union[str, dict]:
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


def parse_tool_args_if_needed(message: ChatMessage) -> ChatMessage:
    if message.tool_calls is not None:
        for tool_call in message.tool_calls:
            tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
    return message


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> Dict:
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: List[str]) -> str:
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: List[Dict[str, str]],
    role_conversions: Dict[MessageRole, MessageRole] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> List[Dict[str, str]]:
    """
    Subsequent messages with the same role will be concatenated to a single message.
    output_message_list is a list of messages that will be used to generate the final message that is chat template compatible with transformers LLM chat template.

    Args:
        message_list (`list[dict[str, str]]`): List of chat messages.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`): Whether to convert images to image URLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:

        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")
        if role in role_conversions:
            role = role_conversions[role]

        if role == "system" and not flatten_messages_as_text:
            message["content"] = message["content"][0]["text"]
        if isinstance(message["content"], list):
            for element in message["content"]:
                if element["type"] == "image" and not is_url(element["image"]):
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {"url": make_image_url(encode_image_base64(element.pop("image")))},
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if len(output_message_list) > 0 and role == output_message_list[-1]["role"]:
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += "\n\n" + message["content"][0]["text"]
            else:
                # TODO handle case where first entry is not text
                text = output_message_list[-1]["content"][0]["text"]
                text += "\n\n" + message["content"][0]["text"]
                output_message_list[-1]["content"][0]["text"] = text
        else:
            if flatten_messages_as_text:
                content = message["content"][0]["text"]
            else:
                content = message["content"]
            output_message_list.append({"role": role, "content": content})

    return output_message_list


class Model:
    def __init__(self, **kwargs):
        self.last_input_token_count = None
        self.last_output_token_count = None
        self.kwargs = kwargs

    def _prepare_completion_kwargs(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        guide: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        convert_images_to_image_urls: bool = False,
        flatten_messages_as_text: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, guide, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        messages = get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )

        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if guide is not None:
            completion_kwargs["guide"] = guide

        # Handle tools parameter
        if tools_to_call_from:
            completion_kwargs.update(
                {
                    "tools": [get_tool_json_schema(tool) for tool in tools_to_call_from],
                    "tool_choice": "auto",
                }
            )

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        return completion_kwargs

    def get_token_counts(self) -> Dict[str, int]:
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        guide: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        """Process the input messages and return the model's response.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            guide (`str`, *optional*):
                The guide or formatting structure to use in the model's response.
            tools_to_call_from (`List[Tool]`, *optional*):
                A list of tools that the model can use to generate responses.
            **kwargs:
                Additional keyword arguments to be passed to the underlying model.

        Returns:
            `ChatMessage`: A chat message object containing the model's response.
        """
        pass  # To be implemented in child classes!

    def to_dict(self) -> Dict:
        """
        Converts the model into a JSON-compatible dictionary.
        """
        model_dictionary = {
            **self.kwargs,
            "last_input_token_count": self.last_input_token_count,
            "last_output_token_count": self.last_output_token_count,
            "model_id": self.model_id,
        }
        for attribute in [
            "custom_role_conversion",
            "temperature",
            "max_tokens",
            "provider",
            "timeout",
            "api_base",
            "torch_dtype",
            "device_map",
            "organization",
            "project",
            "azure_endpoint",
        ]:
            if hasattr(self, attribute):
                model_dictionary[attribute] = getattr(self, attribute)

        dangerous_attributes = ["token", "api_key"]
        for attribute_name in dangerous_attributes:
            if hasattr(self, attribute_name):
                print(
                    f"For security reasons, we do not export the `{attribute_name}` attribute of your model. Please export it manually."
                )
        return model_dictionary

    @classmethod
    def from_dict(cls, model_dictionary: Dict[str, Any]) -> "Model":
        model_instance = cls(
            **{
                k: v
                for k, v in model_dictionary.items()
                if k not in ["last_input_token_count", "last_output_token_count"]
            }
        )
        model_instance.last_input_token_count = model_dictionary.pop("last_input_token_count", None)
        model_instance.last_output_token_count = model_dictionary.pop("last_output_token_count", None)
        return model_instance


class BaseMLXLogitsProcessor(abc.ABC):
    """Enables the model to produce structured output through guide or regex.

    This base class is meant to provide a structure for using logits processor libraries with MLX models.
    """

    @abc.abstractmethod
    def __init__(self, guide: str, tokenizer: "mlx_lm.tokenizer_utils.TokenizerWrapper"):
        pass

    @abc.abstractmethod
    def __call__(self, input_ids: "mx.array", logits: "mx.array") -> "mx.array":
        pass


class MLXLModel(Model):
    """A class to interact with language models loaded using MLX on Apple silicon.

    > [!TIP]
    > You must have `mlx-lm` installed on your machine. Please run `pip install guided_agents[mlx-lm]` if it's not the case.

    Parameters:
        model_id (str):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        tool_name_key (str):
            The key, which can usually be found in the model's chat template, for retrieving a tool name.
        tool_arguments_key (str):
            The key, which can usually be found in the model's chat template, for retrieving tool arguments.
        trust_remote_code (bool):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        logits_processor (BaseMLXLogitsProcessor *optional*):
            Structures model output based on a guide argument to the model's call method.
        kwargs (dict, *optional*):
            Any additional keyword arguments that you want to use in model.generate(), for instance `max_tokens`.

    Example:
    ```python
    >>> engine = MLXModel(
    ...     model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
    ...     max_tokens=10000,
    ... )
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             {"type": "text", "text": "Explain quantum mechanics in simple terms."}
    ...         ]
    ...     }
    ... ]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        trust_remote_code: bool = False,
        logits_processor: Optional[BaseMLXLogitsProcessor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not _is_package_available("mlx_lm"):
            raise ModuleNotFoundError(
                "Please install 'mlx-lm' extra to use 'MLXModel': `pip install 'guided_agents[mlx-lm]'`"
            )
        import mlx_lm

        self.model_id = model_id
        self.model, self.tokenizer = mlx_lm.load(model_id, tokenizer_config={"trust_remote_code": trust_remote_code})
        self.stream_generate = mlx_lm.stream_generate
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.logits_processor = logits_processor

    def _to_message(self, text, tools_to_call_from, finish_reason=None):
        tool_call_start = "Action:\n{"
        if tools_to_call_from and tool_call_start in text:
            # solution for extracting tool JSON without assuming a specific model output format
            maybe_json = "{" + text.split(tool_call_start, 1)[-1][::-1].split("}", 1)[-1][::-1] + "}"
            try:
                content = ""
                parsed_text = json_repair.loads(maybe_json)
            except json.JSONDecodeError as e:
                content = f"\n\nTool JSON decode failure: {e}\n\n"
                content += maybe_json
                parsed_text = {}
            finally:
                tool_name = parsed_text.get(self.tool_name_key, None)
                tool_arguments = parsed_text.get(self.tool_arguments_key, None)
                if tool_name:
                    return ChatMessage(
                        role="assistant",
                        content=content,
                        tool_calls=[
                            ChatMessageToolCall(
                                id=str(uuid.uuid4()),
                                type="function",
                                function=ChatMessageToolCallDefinition(name=tool_name, arguments=tool_arguments),
                            )
                        ],
                    )
        return ChatMessage(role="assistant", content=text, finish_reason=finish_reason)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        guide: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            flatten_messages_as_text=True,  # mlx-lm doesn't support vision models
            messages=messages,
            stop_sequences=stop_sequences,
            guide=guide,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        # completion_kwargs post-process steps needed for mlx-lm
        messages = completion_kwargs.pop("messages")
        prepared_stop_sequences = completion_kwargs.pop("stop", [])
        completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)
        guide = completion_kwargs.pop("guide", None)
        if guide:
            if self.logits_processor is None:
                raise ValueError("Please initialize the model with a logits processor to use a guide.")
            completion_kwargs["logits_processors"] = [self.logits_processor(guide=guide, model_id=self.model_id)]

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        self.last_input_token_count = len(prompt_ids)
        self.last_output_token_count = 0
        text = ""

        for _ in self.stream_generate(self.model, self.tokenizer, prompt=prompt_ids, **completion_kwargs):
            self.last_output_token_count += 1
            text += _.text
            for stop_sequence in prepared_stop_sequences + [self.tokenizer.eos_token]:
                stop_sequence_start = text.rfind(stop_sequence)
                if stop_sequence_start != -1:
                    text = text[:stop_sequence_start]
                    return self._to_message(text, tools_to_call_from, finish_reason="stop_sequence")

        return self._to_message(text, tools_to_call_from, finish_reason=_.finish_reason)


class MLXVLModel(Model):
    """A class to interact with vision language models loaded using MLX on Apple silicon.

    > [!TIP]
    > You must have `mlx-vlm` installed on your machine. Please run `pip install guided_agents[mlx-vlm]` if it's not the case.

    Parameters:
        model_id (str):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        tool_name_key (str):
            The key, which can usually be found in the model's chat template, for retrieving a tool name.
        tool_arguments_key (str):
            The key, which can usually be found in the model's chat template, for retrieving tool arguments.
        trust_remote_code (bool):
            Some models on the Hub require running remote code: for this model, you would have to set this flag to True.
        logits_processor (BaseMLXLogitsProcessor *optional*):
            Structures model output based on a guide argument to the model's call method.
        kwargs (dict, *optional*):
            Any additional keyword arguments that you want to use in model.generate(), for instance `max_tokens`.

    Example:
    ```python
    >>> engine = MLXModel(
    ...     model_id="mlx-community/Qwen2.5-VL-32B-Instruct-4bit",
    ...     max_tokens=10000,
    ... )
    >>> messages = [
    ...     {
    ...         "role": "user",
    ...         "content": [
    ...             {"type": "text", "text": "Explain quantum mechanics in simple terms."}
    ...         ]
    ...     }
    ... ]
    >>> response = engine(messages, stop_sequences=["END"])
    >>> print(response)
    "Quantum mechanics is the branch of physics that studies..."
    ```
    """

    def __init__(
        self,
        model_id: str,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        trust_remote_code: bool = False,
        logits_processor: Optional[BaseMLXLogitsProcessor] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not _is_package_available("mlx_vlm"):
            raise ModuleNotFoundError(
                "Please install 'mlx-vlm' extra to use 'MLXVLModel': `pip install 'guided_agents[mlx-vlm]'`"
            )
        import mlx_vlm

        self.model_id = model_id
        self.model, self.processor = mlx_vlm.load(model_id, processor_kwargs={"trust_remote_code": trust_remote_code})
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key
        self.logits_processor = logits_processor

    def _to_message(self, text, tools_to_call_from, finish_reason=None):
        tool_call_start = "Action:\n{"
        if tools_to_call_from and tool_call_start in text:
            # solution for extracting tool JSON without assuming a specific model output format
            maybe_json = "{" + text.split(tool_call_start, 1)[-1][::-1].split("}", 1)[-1][::-1] + "}"
            try:
                content = text
                parsed_text = json_repair.loads(maybe_json)
            except json.JSONDecodeError as e:
                content = f"\n\nTool JSON decode failure: {e}\n\n"
                content += maybe_json
                parsed_text = {}
            finally:
                tool_name = parsed_text.get(self.tool_name_key, None)
                tool_arguments = parsed_text.get(self.tool_arguments_key, None)
                if tool_name:
                    return ChatMessage(
                        role="assistant",
                        content=content,
                        tool_calls=[
                            ChatMessageToolCall(
                                id=str(uuid.uuid4()),
                                type="function",
                                function=ChatMessageToolCallDefinition(name=tool_name, arguments=tool_arguments),
                            )
                        ],
                    )
        return ChatMessage(role="assistant", content=text, finish_reason=finish_reason)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        guide: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            guide=guide,
            tools_to_call_from=tools_to_call_from,
            **kwargs,
        )
        # completion_kwargs post-process steps needed for mlx-vlm
        messages = completion_kwargs.pop("messages")
        prepared_stop_sequences = completion_kwargs.pop("stop", [])
        completion_kwargs.pop("tools", None)
        completion_kwargs.pop("tool_choice", None)
        guide = completion_kwargs.pop("guide", None)
        if guide:
            if self.logits_processor is None:
                raise ValueError("Please initialize the model with a logits processor to use a guide.")
            completion_kwargs["logits_processors"] = [self.logits_processor(guide=guide, model_id=self.model_id)]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images = []
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image":
                    image = content["image"]
                    if isinstance(image, str) and image.startswith("file://"):
                        image = image[len("file://"):]
                    images.append(image)

        text = ""
        for _ in self.stream_generate(
            self.model,
            self.processor,
            prompt,
            images,
            **completion_kwargs
        ):
            text += _.text
            for stop_sequence in prepared_stop_sequences:
                stop_sequence_start = text.rfind(stop_sequence)
                if stop_sequence_start != -1:
                    text = text[:stop_sequence_start]
                    return self._to_message(text, tools_to_call_from, finish_reason="stop_sequence")

        return self._to_message(text, tools_to_call_from, finish_reason=None)

    def generate_step(
        self,
        input_ids: mx.array,
        model: nn.Module,
        pixel_values,
        mask,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        repetition_penalty: Optional[float] = None,
        repetition_context_size: Optional[int] = 20,
        top_p: float = 1.0,
        logit_bias: Optional[Dict[int, float]] = None,
        logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
        **kwargs,
    ) -> "mlx_vlm.utils.Generator[Tuple[mx.array, mx.array], None, None]":
        """
        A generator producing token ids based on the given prompt from the model.

        Args:
            ...
            logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
            A list of functions that take tokens and logits and return the processed
            logits. Default: ``None``.

        Yields:
            Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
            one token and a vector of log probabilities.
        """
        from mlx_vlm.utils import KVCache

        def sample(logits: mx.array) -> Tuple[mx.array, float]:
            if logit_bias:
                indices = mx.array(list(logit_bias.keys()))
                values = mx.array(list(logit_bias.values()))
                logits[:, indices] += values
            logprobs = logits - mx.logsumexp(logits)

            if temperature == 0:
                token = mx.argmax(logits, axis=-1)
            else:
                if top_p > 0 and top_p < 1.0:
                    token = top_p_sampling(logits, top_p, temperature)
                else:
                    token = mx.random.categorical(logits * (1 / temperature))

            return token, logprobs

        if repetition_penalty and (
            repetition_penalty < 0 or not isinstance(repetition_penalty, float)
        ):
            raise ValueError(
                f"repetition_penalty must be a non-negative float, got {repetition_penalty}"
            )

        y = input_ids
        if hasattr(model.language_model, "make_cache"):
            cache = model.language_model.make_cache()
        else:
            kv_heads = (
                [model.language_model.n_kv_heads] * len(model.language_model.layers)
                if isinstance(model.language_model.n_kv_heads, int)
                else model.language_model.n_kv_heads
            )
            if model.config.model_type == "florence2":
                cache = [
                    (SimpleKVCache(), SimpleKVCache()) for n in model.language_model.layers
                ]
            else:
                cache = [KVCache() for n in kv_heads]

        repetition_context = input_ids.reshape(-1).tolist()

        if repetition_context_size:
            repetition_context = repetition_context[-repetition_context_size:]

        tokens = None
        def _step(y, **kwargs):
            nonlocal repetition_context
            if "decoder_input_ids" in kwargs:
                outputs = model.language_model(
                    cache=cache,
                    **kwargs,
                )
            else:
                outputs = model.language_model(
                    y[None],
                    cache=cache,
                    **kwargs,
                )

            logits = outputs.logits[:, -1, :]

            if logits_processors:
                nonlocal tokens
                tokens = mx.concat([tokens, y]) if tokens is not None else y
                for processor in logits_processors:
                    logits = processor(tokens, logits)

            if repetition_penalty:
                logits = apply_repetition_penalty(
                    logits, repetition_context, repetition_penalty
                )
                y, logprobs = sample(logits)
                repetition_context.append(y.item())
            else:
                y, logprobs = sample(logits)

            if repetition_context_size:
                if len(repetition_context) > repetition_context_size:
                    repetition_context = repetition_context[-repetition_context_size:]
            return y, logprobs.squeeze(0)

        outputs = model(input_ids, pixel_values, cache=cache, mask=mask, **kwargs)

        logits = outputs.logits[:, -1, :]
        y, logprobs = sample(logits)
        mx.async_eval(y, logprobs)

        if outputs.cross_attention_states is not None:
            kwargs = {
                k: v
                for k, v in zip(
                    ["cross_attention_states"], [outputs.cross_attention_states]
                )
            }
        elif outputs.encoder_outputs is not None:
            kwargs = {
                "decoder_input_ids": y[None],
                "encoder_outputs": outputs.encoder_outputs,
            }
        else:
            kwargs = {}

        n = 0
        while True:
            if n != max_tokens:
                next_y, next_logprobs = _step(y, **kwargs)
                mx.async_eval(next_y, next_logprobs)
                y, logprobs = next_y, next_logprobs
            if n == 0:
                mx.eval(y)
            if n == max_tokens:
                break
            if "decoder_input_ids" in kwargs:
                kwargs["decoder_input_ids"] = next_y[None]
            yield y.item(), logprobs
            y, logprobs = next_y, next_logprobs
            n += 1


    def stream_generate(
        self,
        model: nn.Module,
        processor: "mlx_vlm.utils.PreTrainedTokenizer",
        prompt: str,
        image: Union[str, List[str]] = None,
        **kwargs,
    ) -> "Union[str, mlx.utils.Generator[str, None, None]]":
        """
        A generator producing text based on the given prompt from the model.

        Args:
            model (nn.Module): The model to use for generation.
            processor: ...
            prompt (mx.array): The input prompt.
            image: ...
            kwargs: The remaining options get passed to :func:`generate_step`.
            See :func:`generate_step` for more details.

        Yields:
            Generator[Tuple[mx.array, mx.array]]: A generator producing text.
        """
        from mlx_vlm.utils import prepare_inputs

        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        add_special_tokens = not hasattr(processor, "chat_template")
        prompt_tokens = mx.array(
            tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        )

        resize_shape = kwargs.pop("resize_shape", None)
        image_token_index = getattr(model.config, "image_token_index", None)

        if kwargs.get("pixel_values") is None:
            if not image:
                input_ids = prompt_tokens[None, :]
                pixel_values = mask = None
            else:
                inputs = prepare_inputs(
                    processor, image, prompt, image_token_index, resize_shape
                )
                input_ids = inputs["input_ids"]
                pixel_values = inputs["pixel_values"]
                mask = inputs["attention_mask"]
                data_kwargs = {
                    k: v
                    for k, v in inputs.items()
                    if k not in ["input_ids", "pixel_values", "attention_mask"]
                }
                kwargs.update(data_kwargs)
        else:
            input_ids = kwargs.pop("input_ids")
            pixel_values = kwargs.pop("pixel_values")
            mask = kwargs.pop("mask")

        detokenizer = processor.detokenizer
        detokenizer.reset()
        tic = time.perf_counter()
        for n, (token, logprobs) in enumerate(
            self.generate_step(input_ids, model, pixel_values, mask, **kwargs)
        ):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = input_ids.size / prompt_time
                tic = time.perf_counter()

            if token == tokenizer.eos_token_id:
                break

            detokenizer.add_token(token)

            # Yield the last segment if streaming
            yield MLXVLMGenerationResult(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=input_ids.size,
                generation_tokens=n + 1,
                prompt_tps=prompt_tps,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
            )

        detokenizer.finalize()
        yield MLXVLMGenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=input_ids.size,
            generation_tokens=n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop" if token == tokenizer.eos_token_id else "length",
        )


class OpenAIServerModel(Model):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] | None = None,
        project: Optional[str] | None = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        custom_role_conversions: Optional[Dict[str, str]] = None,
        tool_name_key: str = "name",
        tool_arguments_key: str = "arguments",
        **kwargs,
    ):
        try:
            import openai
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install 'openai' extra to use OpenAIServerModel: `pip install 'guided_agents[openai]'`"
            ) from None

        super().__init__(**kwargs)
        self.model_id = model_id
        self.client = openai.OpenAI(
            base_url=api_base,
            api_key=api_key,
            organization=organization,
            project=project,
            **(client_kwargs or {}),
        )
        self.custom_role_conversions = custom_role_conversions
        self.tool_name_key = tool_name_key
        self.tool_arguments_key = tool_arguments_key

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        guide: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            #flatten_messages_as_text=True,
            messages=messages,
            stop_sequences=stop_sequences,
            guide=guide,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        extra_body={}
        guide = completion_kwargs.pop("guide", None)
        if guide:
            extra_body["guided_regex"] = guide
        if stop_sequences:
            extra_body["stop"] = stop_sequences
        if extra_body:
            completion_kwargs["extra_body"] = extra_body
        completion_kwargs.pop("tool_choice", None)
        completion_kwargs.pop("tools", None)

        for attempt in tenacity.Retrying(
            retry=tenacity.retry_if_exception_type((httpx.RemoteProtocolError, openai.NotFoundError)),
            stop=tenacity.stop_after_attempt(6),
            wait=tenacity.wait_random_exponential(multiplier=2, max=60)
        ):
            with attempt:
                response = self.client.chat.completions.create(**completion_kwargs, stream=True)
                response.response.raise_for_status()
                text = ""
                for chunk in response:
                    _ = chunk.choices[0].delta.content
                    #print(_, end="", flush=True)
                    text += _
                finish_reason = chunk.choices[0].finish_reason

        #self.last_input_token_count = response.usage.prompt_tokens
        #self.last_output_token_count = response.usage.completion_tokens
        tool_call_start = "Action:\n{"
        if tools_to_call_from and tool_call_start in text:
            content = text
            try:
                # solution for extracting tool JSON without assuming a specific model output format
                maybe_json = "{" + text.split(tool_call_start, 1)[-1][::-1].split("}", 1)[-1][::-1] + "}"
                parsed_text = json_repair.loads(maybe_json)
            except json.JSONDecodeError as e:
                content += f"\n\nTool call failure: {e}"
                parsed_text = {}
            finally:
                tool_name = parsed_text.get(self.tool_name_key, None)
                tool_arguments = parsed_text.get(self.tool_arguments_key, None)
                if tool_name:
                    return ChatMessage(
                        role="assistant",
                        content=content,
                        tool_calls=[
                            ChatMessageToolCall(
                                id=str(uuid.uuid4()),
                                type="function",
                                function=ChatMessageToolCallDefinition(name=tool_name, arguments=tool_arguments),
                            )
                        ],
                    )
        return ChatMessage(role="assistant", content=text, finish_reason=finish_reason)


try:
    from langchain_community.llms.vllm import VLLMOpenAI
except ImportError:
    # Create a stub class if langchain_community is not available
    class VLLMOpenAI:
        def __init__(self, **kwargs):
            raise ImportError(
                "Please install 'langchain-community' to use GuidedVLLMOpenAI: "
                "`pip install langchain-community`"
            )


class GuidedVLLMOpenAI(VLLMOpenAI):
    """A model that extends VLLMOpenAI to support guide classes for structured generation.

    This class inherits from VLLMOpenAI and adds support for guide classes that define
    structured output patterns. It can be used with langchain react agents and other
    langchain components.

    Parameters:
        model_id (str): The model identifier to use on the vLLM server.
        api_base (str, optional): The base URL of the vLLM server.
        api_key (str, optional): The API key for authentication.
        **kwargs: Additional arguments passed to VLLMOpenAI.

    Example:
        >>> from guided_agents.models import GuidedVLLMOpenAI
        >>> from example.guides import RegexReasoningGuide
        >>>
        >>> model = GuidedVLLMOpenAI(
        ...     model_id="Qwen/Qwen3-8B",
        ...     api_base="http://localhost:8000/v1",
        ...     api_key="your-api-key"
        ... )
        >>>
        >>> guide = RegexReasoningGuide(act="Final Answer: .+")
        >>> response = model.invoke("What is 2+2?", guide=guide)
        >>> print(response)
    """

    def __init__(self, model_id: str, api_base: Optional[str] = None, api_key: Optional[str] = None, **kwargs):
        # Initialize the parent VLLMOpenAI model
        super().__init__(
            model=model_id,
            base_url=api_base,
            api_key=api_key,
            **kwargs
        )

    def _detect_guide_type(self, guide_str: str) -> str:
        """Detect whether a guide string is regex or grammar format.

        Args:
            guide_str: The guide string to analyze

        Returns:
            Either 'guided_regex' or 'guided_grammar' based on the guide format
        """
        # Strong indicators of Lark grammar syntax
        strong_grammar_indicators = [
            'start:', '?start:', 'rule:', '?rule:',  # Lark grammar rules
            ' NL ', 'NL:', 'NL\n',  # Common Lark non-terminal usage
            ': /',  # Rule definition with regex inside
            'paragraph:', 'sentence:', 'reason:',  # Common rule names in Lark guides
        ]
        
        # Check for strong grammar indicators first
        if any(indicator in guide_str for indicator in strong_grammar_indicators):
            return 'guided_grammar'
        
        # If it has multiple actual lines (not escaped \n) with grammar-like structure
        lines = [line.strip() for line in guide_str.split('\n') if line.strip()]
        if len(lines) > 1:
            # Count lines that look like grammar rules (have : followed by something other than escaped chars)
            rule_lines = 0
            for line in lines:
                if ':' in line and not line.startswith('//'):
                    # Check if it's a rule definition, not just regex with colon
                    colon_pos = line.find(':')
                    if colon_pos > 0:
                        rule_name = line[:colon_pos].strip()
                        # Rule names are typically single words or have underscores
                        if rule_name.replace('_', '').replace('?', '').isalpha():
                            rule_lines += 1
            
            # If we have multiple lines that look like grammar rules, it's grammar
            if rule_lines >= 2:
                return 'guided_grammar'
        
        # Otherwise, treat as regex
        return 'guided_regex'

    def invoke(
        self,
        input: Union[str, Any],
        config: Optional[Any] = None,
        *,
        guide: Optional[Union[str, Any]] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        """Invoke the model with optional guide support.

        Args:
            input: The input text or prompt value.
            config: Optional configuration for the run.
            guide: Either a guide string or a guide class instance that has a __call__ method.
            stop: Optional list of stop sequences.
            **kwargs: Additional keyword arguments.

        Returns:
            The model's response as a string.
        """
        # Process guide parameter
        guide_str = None
        if guide is not None:
            if isinstance(guide, str):
                guide_str = guide
            elif hasattr(guide, '__call__'):
                # Guide class instance - call it to get the guide string
                guide_str = guide()
            else:
                raise ValueError(
                    f"Guide must be a string or an object with a __call__ method, got {type(guide)}"
                )

        # Prepare the request with guide information
        if guide_str:
            # Detect guide type and set appropriate parameter
            guide_param = self._detect_guide_type(guide_str)

            # Add guided generation parameters via extra_body without modifying model state
            extra_body = kwargs.setdefault('extra_body', {})
            extra_body[guide_param] = guide_str

        # Call the parent's invoke method
        return super().invoke(input, config=config, stop=stop, **kwargs)




__all__ = [
    "MessageRole",
    "tool_role_conversions",
    "get_clean_message_list",
    "Model",
    "MLXLModel",
    "MLXVLModel",
    "BaseMLXLogitsProcessor",
    "OpenAIServerModel",
    "GuidedVLLMOpenAI",
    "ChatMessage",
]
