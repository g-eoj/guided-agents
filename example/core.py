# Copyright 2025 g-eoj
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


import enum
import datetime
import logging
import numpy
import os
import pathlib

from models import model
from tools import *

from guided_agents import CodeAgent, ToolCallingAgent


logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class MaxSteps(enum.IntEnum):
    ONE=1
    FEW=3
    SOME=5
    MANY=8


class ThinkingEffort(enum.IntEnum):
    UNDERTHINK_3=1
    UNDERTHINK_2=2
    UNDERTHINK_1=3
    NORMAL=5
    OVERTHINK_1=8
    OVERTHINK_2=13
    OVERTHINK_3=21


code_starter = r"""
import docx
import numpy
import openpyxl
import pandas
import pptx
import re


# Check if a file_path variable has been given
try:
    print\(file_path\)
except:
    pass
"""

def run_agents(question, file_path, log_dir, model=model, strict_answers=False):

    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(log_dir)
    logger.propagate = False
    f = logging.Formatter(
        fmt="[ %(agent)s | %(stage)s | %(model)s | %(asctime)s ]\n%(message)s\n\n",
        datefmt='%Y-%m-%d %I:%M:%S%p',
    )
    h = logging.FileHandler(
        filename=log_dir + "/log.txt",
    )
    h.setFormatter(f)
    logger.addHandler(h)

    additional_args = {}
    images = []
    if file_path:
        if pathlib.Path(file_path).suffix in [".jpg", ".png"]:
            images = [file_path]
        else:
            additional_args = {"file_path": file_path}

    if strict_answers:
        final_answer_requirements = (
            "\n\nYour final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings."
            " If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise."
            " If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise."
            " If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
        )
    else:
        final_answer_requirements = "\n\nBe nice."

    complexity_factor = 1 + numpy.log(len(question.split("."))**0.1)

    online_researcher = ToolCallingAgent(
        name="online_researcher",
        description="Ask this thrall to research a topic online. Do not ask it to solve tasks or read local files. Information obtained from this thrall doesn't need to be verifed.",
        model=model,
        inherit_knowledge=False,
        initial_guide=(
            fr'Thought: ([^\.\n]+?\.){{1,{ThinkingEffort.UNDERTHINK_1}}}\n'
            r'Action:\n\{\n  "name": "[^"\n]+?",\n  "arguments": \["[^\n\}]+?\]\n\}'
        ),
        guide=(
            fr'Thought: Do I use the \'final_answer\' tool now\? ([^\.\n]+?\.){{1,{ThinkingEffort.UNDERTHINK_1}}}\n'
            r'Action:\n\{\n  "name": "[^"\n]+?",\n  "arguments": \["[^\n\}]+?\]\n\}'
        ),
        final_guide=(
            fr'Thought: I must use the \'final_answer\' tool now\. ([^\.\n]+?\.){{1,{ThinkingEffort.UNDERTHINK_2}}}\n'
            r'Action:\n\{\n  "name": "final_answer",\n  "arguments": \["[^\n\}]+?\]\n\}'
        ),
        tools=[
            GoogleSearchTool(api_key=os.environ["SERPER_API_TOKEN"]),
            GoogleScholarSearchTool(api_key=os.environ["SERPER_API_TOKEN"]),
            WebReader(model),
        ],
        max_steps=MaxSteps.SOME,
        verbosity_level=-1,
        logger=logger,
    )
    python_coder = CodeAgent(
        name="python_coder",
        description="Call this thrall to use Python to compute something or open local files.",
        model=model,
        inherit_knowledge=True,
        initial_guide=(
            fr'Thought: First I\'ll check the file_path variable\.\n'
            fr'Code:\n```(?:py|python)?{code_starter}```<end_code>'
        ),
        guide=(
            fr'Thought: Do I use the \'final_answer\' tool now\? ([^\.\n]+?\.){{1,{ThinkingEffort.UNDERTHINK_1}}}\n'
            r'Code:\n```(?:py|python)?\n[^`]+?\n```<end_code>'
        ),
        final_guide=(
            fr'Thought: I must use the \'final_answer\' tool now\. ([^\.\n]+?\.){{1,{ThinkingEffort.UNDERTHINK_2}}}\n'
            r'Code:\n```(?:py|python)?\n[^`]+?\n```<end_code>'
        ),
        additional_authorized_imports=["docx", "numpy", "openpyxl", "pandas", "pptx", "re"],
        tools=[],
        max_steps=MaxSteps.SOME,
        verbosity_level=-1,
        logger=logger,
    )
    brain = ToolCallingAgent(
        name="brain",
        model=model,
        initial_guide=(
            fr'Thought: ([^\.\n]+?\.){{1,{ThinkingEffort.NORMAL}}}\n'
            r'Action:\n\{\n  "name": "[^"\n]+?",\n  "arguments": \["[^\n\}]+?\]\n\}'
        ),
        guide=(
            fr'Thought: Do I use the \'final_answer\' tool now\? ([^\.\n]+?\.){{1,{ThinkingEffort.UNDERTHINK_1}}}\n'
            r'Action:\n\{\n  "name": "[^"\n]+?",\n  "arguments": \["[^\n\}]+?\]\n\}'
        ),
        final_guide=(
            fr'Thought: I must use the \'final_answer\' tool now\. ([^\.\n]+?\.){{1,{ThinkingEffort.UNDERTHINK_2}}}\n'
            r'Action:\n\{\n  "name": "final_answer",\n  "arguments": \["[^\n\}]+?\]\n\}'
        ),
        tools=[MLXAudioTranscribe()],
        managed_agents=[online_researcher, python_coder],
        max_steps=int(MaxSteps.SOME * complexity_factor),
        verbosity_level=-1,
        logger=logger,
    )

    help_prompt = (
        "\nIf your task instructions don't make sense, use the 'final_answer' tool to ask for clarification."
    )
    online_researcher.prompt_templates["system_prompt"] += (
        " You search the web for links and then always read the relevant web pages to find information."
        " You always read Wikipedia links first. "
        " Your final answer should be a short sentence or list."
    )
    brain.prompt_templates["system_prompt"] += final_answer_requirements

    answer = brain.run(question, additional_args=additional_args, images=images)
    return answer
