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
import importlib
import inspect
import time
from collections import deque
from logging import getLogger
from typing import Any, Callable, Dict, Generator, List, Optional, TypedDict, Union

import yaml
from jinja2 import StrictUndefined, Template

from .agent_types import AgentAudio, AgentImage, AgentType, handle_agent_output_types
from .default_tools import FinalAnswerTool
from .local_python_executor import (
    BASE_BUILTIN_MODULES,
    LocalPythonExecutor,
    fix_final_answer_code,
)
from .memory import ActionStep, AgentMemory, SystemPromptStep, TaskStep, ToolCall
from .models import (
    ChatMessage,
)
from .monitoring import (
    AgentLogger,
    LogLevel,
    Monitor,
)
from .tools import Tool
from .utils import (
    AgentError,
    parse_code_blobs,
    parse_json_tool_call,
)


logger = getLogger(__name__)


def populate_template(template: str, variables: Dict[str, Any]) -> str:
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(f"Error during jinja template rendering: {type(e).__name__}: {e}")


class ManagedAgentPromptTemplate(TypedDict):
    """
    Prompt templates for the managed agent.

    Args:
        task (`str`): Task prompt.
        report (`str`): Report prompt.
    """

    task: str
    report: str


class PromptTemplates(TypedDict):
    """
    Prompt templates for the agent.

    Args:
        system_prompt (`str`): System prompt.
        managed_agent ([`~agents.ManagedAgentPromptTemplate`]): Managed agent prompt templates.
    """

    system_prompt: str
    managed_agent: ManagedAgentPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
)


class MultiStepAgent:
    """
    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to solve the task.
        tool_parser (`Callable`, *optional*): Function used to parse the tool calls from the LLM output.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the agent's logs.
        final_guide (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        guide (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]`, *optional*): Callbacks that will be called at each step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when called as a managed agent.
        final_answer_checks (`list`, *optional*): List of Callables to run before returning a final answer for checking validity.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        max_steps: int = 20,
        tool_parser: Optional[Callable] = None,
        verbosity_level: LogLevel = LogLevel.INFO,
        guide: Optional[str] = None,
        initial_guide: Optional[str] = None,
        final_guide: Optional[str] = None,
        managed_agents: Optional[List] = None,
        step_callbacks: Optional[List[Callable]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        provide_run_summary: bool = False,
        final_answer_checks: Optional[List[Callable]] = None,
        logger: Optional = None,
        inherit_knowledge: bool = False,
    ):
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        self.max_steps = max_steps
        self.step_number = 0
        self.tool_parser = tool_parser or parse_json_tool_call
        self.guide = guide
        self.initial_guide = initial_guide
        self.final_guide = final_guide
        self.state = {}
        self.name = name
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = final_answer_checks
        self.inherit_knowledge = inherit_knowledge

        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.system_prompt = self.initialize_system_prompt()
        self.input_messages = None
        self.task = None
        self.memory = AgentMemory(self.system_prompt)
        self.logger = AgentLogger(level=verbosity_level, logger=logger)
        self.monitor = Monitor(self.model, self.logger)
        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.step_callbacks.append(self.monitor.update_metrics)

    def _setup_managed_agents(self, managed_agents):
        self.managed_agents = {}
        if managed_agents:
            assert all(agent.name and agent.description for agent in managed_agents), (
                "All managed agents need both a name and a description!"
            )
            self.managed_agents = {agent.name: agent for agent in managed_agents}

    def _setup_tools(self, tools):
        assert all(isinstance(tool, Tool) for tool in tools), "All elements must be instance of Tool (or a subclass)"
        self.tools = {tool.name: tool for tool in tools}
        if "final_answer" not in self.tools:
            self.tools["final_answer"] = FinalAnswerTool()

    def _validate_tools_and_managed_agents(self, tools, managed_agents):
        tool_and_managed_agent_names = [tool.name for tool in tools]
        if managed_agents is not None:
            for agent in managed_agents:
                tool_and_managed_agent_names.append(agent.name)
                for tool in agent.tools.values():
                    if tool.name != "final_answer":
                        tool_and_managed_agent_names.append(tool.name)
        if len(tool_and_managed_agent_names) != len(set(tool_and_managed_agent_names)):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: Optional[List[str]] = None,
        additional_args: Optional[Dict] = None,
        max_steps: Optional[int] = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in a streaming way.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[str]`, *optional*): Paths to image(s).
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.

        Example:
        ```py
        from guided_agents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        if not isinstance(self.task, str):
            raise RuntimeError(self.task)
        if additional_args:
            self.task += f"\n\nThe user has shared these variables for you to access in Python:\n\n{additional_args}"
            self.state.update(additional_args)
        self.system_prompt = self.initialize_system_prompt()
        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )

        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run(task=self.task, max_steps=max_steps, images=images)
        # Outputs are returned only at the end. We only look at the last step.
        return deque(self._run(task=self.task, max_steps=max_steps, images=images), maxlen=1)[0]

    def _run(
        self, task: str, max_steps: int, images: List[str] | None = None,
    ) -> Generator[ActionStep | AgentType, None, None]:
        final_answer = None
        self.step_number = 1
        while final_answer is None and self.step_number <= max_steps:
            step_start_time = time.time()
            is_final = self.step_number == max_steps
            memory_step = ActionStep(
                step_number=self.step_number,
                start_time=step_start_time,
                observations_images=images,
                is_final=is_final
            )
            try:
                final_answer = self._execute_step(task, memory_step)
            except AgentError as e:
                memory_step.error = e
            finally:
                self._finalize_step(memory_step, step_start_time)
                yield memory_step
                self.step_number += 1

        self.logger.log_answer(
            str(final_answer),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            title=self.name if hasattr(self, "name") else None,
        )
        yield handle_agent_output_types(final_answer)

    def _execute_step(self, task: str, memory_step: ActionStep) -> Union[None, Any]:
        final_answer = self.step(memory_step)
        if final_answer is not None and self.final_answer_checks:
            self._validate_final_answer(final_answer)
        return final_answer

    def _validate_final_answer(self, final_answer: Any):
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(f"Check {check_function.__name__} failed with error: {e}", self.logger)

    def _finalize_step(self, memory_step: ActionStep, step_start_time: float):
        memory_step.end_time = time.time()
        memory_step.duration = memory_step.end_time - step_start_time
        self.memory.steps.append(memory_step)
        for callback in self.step_callbacks:
            # For compatibility with old callbacks that don't take the agent as an argument
            callback(memory_step) if len(inspect.signature(callback).parameters) == 1 else callback(
                memory_step, agent=self
            )

    def initialize_system_prompt(self):
        """To be implemented in child classes"""
        pass

    def memory_to_messages(self) -> List[Dict[str, str]]:
        messages = self.memory.system_prompt.to_messages()
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages())
        return messages

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def execute_tool_call(self, tool_name: str, arguments: Union[Dict[str, str], str]) -> Any:
        """
        Execute tool with the provided input and returns the result.
        This method replaces arguments with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the Tool to execute (should be one from self.tools).
            arguments (Dict[str, str]): Arguments passed to the Tool.
        """
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            return f"Unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except:
                pass
        try:
            if isinstance(arguments, str):
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(arguments, additional_args=self.state, memory_steps=self.memory.steps)
                else:
                    observation = available_tools[tool_name].__call__(arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, list):
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(", ".join(arguments), additional_args=self.state, memory_steps=self.memory.steps)
                else:
                    observation = available_tools[tool_name].__call__(*arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and value in self.state:
                        arguments[key] = self.state[value]
                if tool_name in self.managed_agents:
                    observation = available_tools[tool_name].__call__(arguments, additional_args=self.state, memory_steps=self.memory.steps)
                else:
                    observation = available_tools[tool_name].__call__(**arguments, sanitize_inputs_outputs=True)
            else:
                return f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
            return observation
        except Exception as e:
            return f"Tool call failed with error:\n{e}"

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """To be implemented in children classes. Should return either None if the step is not final."""
        pass

    def __call__(self, *args, additional_args=None, memory_steps=None):
        """Adds additional prompting for the managed agent, runs it, and wraps the output.

        This method is called only by a managed agent.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=str(args[0])),
        )
        if self.inherit_knowledge:
            self.memory.steps = memory_steps
        report = self.run(full_task, additional_args=additional_args, reset=False)
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"],
            variables=dict(name=self.name, task=str(args[0]), final_answer=report),
        )
        return answer

    def to_dict(self) -> Dict[str, Any]:
        """Converts agent into a dictionary."""
        # TODO: handle serializing step_callbacks and final_answer_checks
        for attr in ["final_answer_checks", "step_callbacks"]:
            if getattr(self, attr, None):
                self.logger.log(f"This agent has {attr}: they will be ignored by this method.", LogLevel.INFO)

        tool_dicts = [tool.to_dict() for tool in self.tools.values()]
        tool_requirements = {req for tool in self.tools.values() for req in tool.to_dict()["requirements"]}
        managed_agents_requirements = {
            req for managed_agent in self.managed_agents.values() for req in managed_agent.to_dict()["requirements"]
        }
        requirements = tool_requirements | managed_agents_requirements
        if hasattr(self, "authorized_imports"):
            requirements.update(
                {package.split(".")[0] for package in self.authorized_imports if package not in BASE_BUILTIN_MODULES}
            )

        agent_dict = {
            "tools": tool_dicts,
            "model": {
                "class": self.model.__class__.__name__,
                "data": self.model.to_dict(),
            },
            "managed_agents": {
                managed_agent.name: managed_agent.__class__.__name__ for managed_agent in self.managed_agents.values()
            },
            "prompt_templates": self.prompt_templates,
            "max_steps": self.max_steps,
            "verbosity_level": int(self.logger.level),
            "guide": self.guide,
            "name": self.name,
            "description": self.description,
            "requirements": list(requirements),
        }
        if hasattr(self, "authorized_imports"):
            agent_dict["authorized_imports"] = self.authorized_imports
        if hasattr(self, "use_e2b_executor"):
            agent_dict["use_e2b_executor"] = self.use_e2b_executor
        if hasattr(self, "max_print_outputs_length"):
            agent_dict["max_print_outputs_length"] = self.max_print_outputs_length
        return agent_dict


class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("guided_agents.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            **kwargs,
        )
        self.tool_call_history = {}

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={"tools": self.tools, "managed_agents": self.managed_agents},
        )
        return system_prompt

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        memory_messages = self.memory_to_messages()
        self.input_messages = memory_messages

        # Add new step in logs
        memory_step.model_input_messages = memory_messages.copy()

        additional_args = {}
        if self.guide:
            additional_args["guide"] = self.guide
        if self.step_number ==1 and self.initial_guide:
            additional_args["guide"] = self.initial_guide
        if memory_step.is_final and self.final_guide:
            additional_args["guide"] = self.final_guide
        model_message: ChatMessage = self.model(
            memory_messages,
            tools_to_call_from=list(self.tools.values()),
            **additional_args
        )
        memory_step.model_output_message = model_message
        memory_step.model_output = model_message.content
        self.logger.log_thought(
            model_message.content,
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            title=self.name if hasattr(self, "name") else None,
        )
        if model_message.finish_reason == "length":
            msg = "Your response was too long. Please try again with a much shorter response."
            self.logger.log_observation(
                msg,
                subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
                title=self.name if hasattr(self, "name") else None,
            )
            memory_step.observations = msg
            return None
        if not model_message.tool_calls:
            memory_step.observations = model_message
            self.logger.log_observation(
                model_message.content,
                subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
                title=self.name if hasattr(self, "name") else None,
            )
            return None
        tool_call = model_message.tool_calls[0]
        tool_name, tool_call_id = tool_call.function.name, tool_call.id
        tool_arguments = tool_call.function.arguments
        all_previous_tool_arguments = self.tool_call_history.get(tool_name, [])
        if tool_name != "final_answer" and tool_arguments in all_previous_tool_arguments:
            memory_step.observations = f"Try something different. You already tried '{tool_name}' with arguments: {tool_arguments}"
            self.logger.log_observation(
                memory_step.observations,
                subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
                title=self.name if hasattr(self, "name") else None,
            )
            return None
        all_previous_tool_arguments.append(tool_arguments)
        self.tool_call_history[tool_name] = all_previous_tool_arguments

        memory_step.tool_calls = [ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call_id)]

        if tool_arguments is None:
            tool_arguments = {}
        observation = self.execute_tool_call(tool_name, tool_arguments)
        if tool_name == "final_answer":
            memory_step.action_output = observation
            return observation
        observation_type = type(observation)
        if observation_type in [AgentImage, AgentAudio]:
            if observation_type == AgentImage:
                observation_name = "image.png"
            elif observation_type == AgentAudio:
                observation_name = "audio.mp3"
            # TODO: observation naming could allow for different names of same type
            self.state[observation_name] = observation
            updated_information = f"Stored '{observation_name}' in memory."
        else:
            updated_information = (
                f"You called the {tool_name} with arguments: {tool_arguments}\n\n"
                f"The answer from the {tool_name} is:\n{str(observation).strip()}"
            )
        self.logger.log_observation(
            updated_information,
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            title=self.name if hasattr(self, "name") else None,
        )
        memory_step.observations = updated_information
        return None


class CodeAgent(MultiStepAgent):
    """
    In this agent, the tool calls will be formulated by the LLM in code format, then parsed and executed.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        guide (`dict[str, str]`, *optional*): Grammar used to parse the LLM output.
        additional_authorized_imports (`list[str]`, *optional*): Additional authorized imports for the agent.
        use_e2b_executor (`bool`, default `False`): Whether to use the E2B executor for remote code execution.
        max_print_outputs_length (`int`, *optional*): Maximum length of the print outputs.
        **kwargs: Additional keyword arguments.

    """

    def __init__(
        self,
        tools: List[Tool],
        model: Callable[[List[Dict[str, str]]], ChatMessage],
        prompt_templates: Optional[PromptTemplates] = None,
        guide: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        max_print_outputs_length: Optional[int] = None,
        **kwargs,
    ):
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self.max_print_outputs_length = max_print_outputs_length
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("guided_agents.prompts").joinpath("code_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            guide=guide,
            **kwargs,
        )
        if "*" in self.additional_authorized_imports:
            self.logger.log(
                "Caution: you set an authorization for all imports, meaning your agent can decide to import any package it deems necessary. This might raise issues if the package is not installed in your environment.",
                0,
            )
        all_tools = {**self.tools, **self.managed_agents}
        self.python_executor = LocalPythonExecutor(
            self.additional_authorized_imports,
            #all_tools,
            max_print_outputs_length=max_print_outputs_length,
        )

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
            },
        )
        return system_prompt

    def step(self, memory_step: ActionStep) -> Union[None, Any]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Returns None if the step is not final.
        """
        # Observation
        memory_messages = self.memory_to_messages()

        self.input_messages = memory_messages.copy()
        memory_step.model_input_messages = memory_messages.copy()

        # Thought
        additional_args = {}
        if self.guide:
            additional_args["guide"] = self.guide
        if self.step_number ==1 and self.initial_guide:
            additional_args["guide"] = self.initial_guide
        if memory_step.is_final and self.final_guide:
            additional_args["guide"] = self.final_guide
        chat_message: ChatMessage = self.model(
            self.input_messages,
            stop_sequences=["<end_code>", "Observation:"],
            **additional_args,
        )
        model_output = chat_message.content

        self.logger.log_thought(
            model_output,
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            title=self.name if hasattr(self, "name") else None,
        )
        if chat_message.finish_reason == "length":
            msg = "Your response was too long. Please try again with a much shorter response."
            self.logger.log_observation(
                msg,
                subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
                title=self.name if hasattr(self, "name") else None,
            )
            memory_step.observations = msg
            return None

        memory_step.model_output_message = chat_message

        # Action
        code_action = fix_final_answer_code(parse_code_blobs(model_output))
        code_action = code_action.replace("final_answer(answer=", "final_answer(")
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(
                code_action,
            )
            observation = "Your Python console:\n" + execution_logs
            sanitized_output = str(output).replace("`", "'")
            observation += "\n\nYour Python code output:\n" + sanitized_output + "\n\n"
        except Exception as e:
            output = None
            observation = "Your Python console:\n\n" + str(e)

        self.logger.log_observation(
            observation,
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            title=self.name if hasattr(self, "name") else None,
        )

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]
        memory_step.observations = observation
        memory_step.action_output = output

        return output if is_final_answer else None
