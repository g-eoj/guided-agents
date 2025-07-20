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


import datetime
import enum
import logging
import os

from guides import *
from models import *
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


def run_agents(question, file_path, log_dir, model=model, strict_answers=False):

    # logging
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

    # baseline thinking effort
    task_complexity = ThinkingEffort.NORMAL
    # increase thinking effort based on task sentence count
    sentence_count = max(min(len(question.split(". ")), 20), 1)
    task_complexity += numpy.log(sentence_count**ThinkingEffort.NORMAL)

    # gaia requirements
    if strict_answers:
        final_answer_requirements = (
            "\n\nYour final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings."
            " If you are asked for a number, don’t use a comma to write your number neither use units such as $ or percent sign unless specified otherwise."
            " If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise."
            " If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
        )
    else:
        final_answer_requirements = "\n\nBe nice."

    # lark acts
    action_act = (
        'act: "Action:" NL "{" NL action_name NL action_args NL "}"\n'
        'action_name: Q "name" Q ": " Q ACTION_NAME Q\n'
        'ACTION_NAME: /[a-z_]+/\n'
        'action_args[lazy]: Q "arguments" Q ": [" Q ACTION_ARGS Q "]"\n'
        'ACTION_ARGS: /.+/\n'
    )
    code_act=(
        'act[lazy]: "Code:" NL "```python" NL CODE+ "```<end_code>"\n'
        'CODE: /[^`\\n]+/ NL\n'
    )
    final_answer_code_act=(
        'act[lazy]: "Code:" NL "```python" NL CODE+ FINAL_ANSWER "```<end_code>"\n'
        'CODE: /[^`\\n]+/ NL\n'
        'FINAL_ANSWER: "final_answer(" /[^`\\n]+/  ")" NL\n'
    )
    note_act=(
        'act: "<not_relevant>" | NOTE\n'
        'NOTE: "Note: " /.+/\n'
    )

    # agent definitions
    online_researcher = ToolCallingAgent(
        name="online_researcher",
        description="Ask this thrall to find information online. Give it detailed instructions about the information you are looking for. Information obtained from this thrall doesn't need to be verifed.",
        model=model,
        inherit_knowledge=False,
        guide=LarkReasoningGuide(
            act=action_act,
            model_id=model.model_id,
            reasoning_paragraph_limit=3,
            reasoning_sentence_limit=5
        )(),
        tools=[
            GoogleSearchTool(api_key=os.environ["SERPER_API_TOKEN"]),
            GoogleScholarSearchTool(api_key=os.environ["SERPER_API_TOKEN"]),
            WebReader(
                model,
                guide=LarkReasoningGuide(
                    act=note_act,
                    model_id=model.model_id,
                    reasoning_paragraph_limit=3,
                    reasoning_sentence_limit=5
                )(),
                max_iterations_per_page=50,
                max_workers=3,
                min_notes_if_possible=3,
                logger=logger
            ),
        ],
        max_steps=MaxSteps.MANY,
        verbosity_level=-1,
        logger=logger,
    )

    coder = CodeAgent(
        name="coder",
        description="Ask this thrall to do computations with Python. Do not give it code, just give it a task. Don't ask it to find information online, it doesn't have access to the internet.",
        model=coder_model,
        inherit_knowledge=False,
        guide=LarkReasoningGuide(
            act=code_act,
            model_id=model.model_id,
            reasoning_paragraph_limit=5,
            reasoning_sentence_limit=5
        )(),
        final_guide=LarkReasoningGuide(
            act=final_answer_code_act,
            model_id=model.model_id,
            reasoning_paragraph_limit=5,
            reasoning_sentence_limit=5
        )(),
        tools=[],
        additional_authorized_imports=["numpy","pandas"],
        max_steps=MaxSteps.FEW,
        verbosity_level=-1,
        logger=logger,
    )

    brain_tools = [NoteToSelf()]
    if file_path:
        brain_tools.append(
            FileReader(
                model,
                guide=LarkReasoningGuide(
                    act=note_act,
                    model_id=model.model_id,
                    reasoning_paragraph_limit=3,
                    reasoning_sentence_limit=5
                )(),
                max_iterations_per_page=50,
                max_workers=2,
                min_notes_if_possible=3,
                logger=logger,
                path=file_path,

            ),
        )
    brain = ToolCallingAgent(
        name="brain",
        model=model,
        guide=LarkReasoningGuide(
            act=action_act,
            model_id=model.model_id,
            reasoning_paragraph_limit=max(1, int(task_complexity // 2)),
            reasoning_sentence_limit=int(task_complexity),
        )(),
        tools=brain_tools,
        managed_agents=[coder, online_researcher],
        max_steps=max(5, int(task_complexity)),
        verbosity_level=-1,
        logger=logger,
    )

    help_prompt = (
        "\nIf your task instructions don't make sense, use the 'final_answer' tool to ask for clarification."
    )
    time_prompt = f"\nToday's date is {datetime.datetime.now():%Y-%m-%d}."

    brain.prompt_templates["system_prompt"] += final_answer_requirements + time_prompt
    coder.prompt_templates["system_prompt"] += "\nTo give your final answer, you must call the provided `final_answer` function in your Python console." + help_prompt + time_prompt
    online_researcher.prompt_templates["system_prompt"] += (
        " You search the internet information."
        " First, you search for links or papers that might be relevant to your task."
        " Then you read a page or paper that fits the query."
        " However, you don't read anything that is homework or a dataset. "
        " Only read one link at a time. "
        " Stop reading once you have found the requested information. "
        " If the links don't seem promising, search again with a different query. "
        " You do not have the ability to watch videos, so if you are asked to find information on a video platform, such as YouTube, read the video description for clues."
    ) + time_prompt

    answer = brain.run(question)
    return answer
