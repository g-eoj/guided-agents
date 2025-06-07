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
import pathlib
import pprint
import shutil
import subprocess
import sys
import tempfile

import numpy
import pdf2image
from file_loaders import *
from models import *
from PIL import Image, ImageChops
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


def get_lo_exe() -> str:
    name = "soffice"
    path = None
    match sys.platform:
        case "win32":
            path = pathlib.Path(os.environ["PROGRAMFILES"]) / "LibreOffice/program"
        case "darwin":
            path = pathlib.Path("/Applications/LibreOffice.app/Contents/MacOS")
    if not (exe := shutil.which(name, path=path)):
        raise FileNotFoundError("LibreOffice not found")
    return exe

def to_images(file_path):
    images = []
    if file_path.endswith("pdf"):
        images += pdf2image.convert_from_path(file_path, dpi=150)
    elif not file_path.endswith("mp3"):
        with tempfile.TemporaryDirectory() as outdir:
            cmd = [get_lo_exe(), "--convert-to", "pdf", "--outdir", outdir, file_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL)
            for pdf in pathlib.Path(outdir).glob("*.pdf"):
                images += pdf2image.convert_from_path(pdf, dpi=150)
    images = [trim(image) for image in images]
    return images

def trim(image):
    bg = Image.new(image.mode, image.size, "white")
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    if bbox:
        bbox = list(bbox)
        bbox[0] = bbox[0] - 14
        bbox[1] = bbox[1] - 14
        bbox[2] = bbox[2] + 14
        bbox[3] = bbox[3] + 28
        return image.crop(bbox)

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
    task_complexity = 1
    # increase thinking effort based on task sentence count
    sentence_count = max(min(len(question.split(". ")), 8), 1)
    task_complexity += numpy.log(sentence_count**ThinkingEffort.UNDERTHINK_1)

    # file processing
    additional_args = {}
    images = []
    file_loaders = []
    if file_path:
        file_name = pathlib.Path(file_path).name
        file_suffix = pathlib.Path(file_path).suffix

        if file_suffix in [".jpg", ".png"]:
            images = [file_path]
        if file_suffix in [".docx"]:
            #additional_args["file_path"] = file_path
            file_loaders.append(DOCXReader(file_path))
            #images = to_images(file_path)
        if file_suffix in [".mp3"]:
            file_loaders.append(MLXAudioTranscribe(file_path))
        if file_suffix in [".pptx"]:
            #additional_args["file_path"] = file_path
            file_loaders.append(PPTXReader(file_path))
            #images = to_images(file_path)
        if file_suffix in [".xlsx"]:
            #additional_args["file_path"] = file_path
            file_loaders.append(ExcelReader(file_path))
            #images = to_images(file_path)
        if file_suffix in [".txt", ".py"]:
            file_loaders.append(TXTReader(file_path))
            #images = to_images(file_path)

        for loader in file_loaders:
            question += f"\n\n{file_name} as text:\n" + pprint.pformat(loader.forward())

    if images:
        question += f"\n\n{file_name} as images:\n"
        # increase thinking effort based on image count
        image_count = min(len(images), 10)
        task_complexity += numpy.log(image_count**ThinkingEffort.NORMAL)

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


    def guide(act='act: /.{0,500}/', paragraph_limit=1, sentence_limit=3):
        core = (
            'start: <[151667]> NL think <[151668]> NL act\n'
            f'think: paragraph{{1,{int(paragraph_limit)}}}\n'
            f'paragraph: sentence{{1,{int(sentence_limit)}}} NL\n'
            'sentence[lazy]: /.+([\\.\\?!]{1}[ \\n]{1})/\n'
            'NL: /\\n/\n'
            'Q: /"/ \n'
        )
        return core + act

    action_act = (
        'act: "Action:" NL "{" NL action_name NL action_args NL\n'
        'action_name: Q "name" Q ":  "  Q ACTION_NAME Q\n'
        'ACTION_NAME: /[a-z_]+/\n'
        'action_args[lazy]: Q "arguments" Q ":  [" Q /.+/ Q "]" NL "}" NL\n'
    )

    code_act=(
        'act: "Code:" NL "```python" NL /[^`]+/ "```<end_code>"\n'
    )

    note_act=(
        'act: "Note:" /.*/\n'
    )

    # agent definitions
    online_researcher = ToolCallingAgent(
        name="online_researcher",
        description="Ask this thrall to find information online. Give it detailed instructions about the information you are looking for. Information obtained from this thrall doesn't need to be verifed.",
        model=model,
        inherit_knowledge=False,
        guide=guide(action_act),
        tools=[
            GoogleSearchTool(api_key=os.environ["SERPER_API_TOKEN"]),
            GoogleScholarSearchTool(api_key=os.environ["SERPER_API_TOKEN"]),
            WebReader(
                model,
                chunk_size=int(3e4),
                guide=guide(note_act, paragraph_limit=2),
                max_iterations_per_page=50,
                max_workers=1,
                min_notes_if_possible=2,
                logger=logger
            ),
        ],
        max_steps=MaxSteps.MANY,
        verbosity_level=-1,
        logger=logger,
    )
    coder = CodeAgent(
        name="coder",
        description="Ask this thrall to do computations with Python. Do not give it code, just give it a task.",
        model=coder_model,
        inherit_knowledge=False,
        guide=guide(code_act),
        #guide=(
        #    r'<think>\nDo I use the \'final_answer\' function now\? '
        #    fr'([^\n\.]+?\.){{1,{ThinkingEffort.NORMAL}}}\n\n'
        #    r'</think>\n'
        #    r'Code:\n```(?:py|python)?\n[^`]+?\n```<end_code>'
        #),
        #final_guide=(
        #    r'<think>\nI must use the \'final_answer\' function now\. '
        #    fr'([^\n\.]+?\.){{1,{ThinkingEffort.NORMAL}}}\n\n'
        #    r'</think>\n'
        #    r'Code:\n```(?:py|python)?\nfinal_answer\("[^`]+?"\)\n```<end_code>'
        #),
        tools=[],
        additional_authorized_imports=["numpy","pandas"],
        max_steps=MaxSteps.SOME,
        verbosity_level=-1,
        logger=logger,
    )
    brain = ToolCallingAgent(
        name="brain",
        model=model,
        guide=guide(
            action_act,
            paragraph_limit=int(task_complexity),
            sentence_limit=int(task_complexity),
        ),
        #initial_guide=(
        #    fr'([^\.\n]+?\.){{0,{int(task_complexity)}}}?'
        #    r'\n<[151668]>\n'
        #    r'Action:\n\{\n  "name": "[^"\n]+?",\n  "arguments": \["[^\]\n\}]+?"\]\n\}'
        #),
        #guide=(
        #    fr'([^\.\n]+?\.){{0,{int(task_complexity)}}}?\n\n'
        #    r'What have I learned from my actions so far\?'
        #    fr'([^\.\n]+?\.){{1,{int(task_complexity)}}}?\n\n'
        #    r'What action do I take now\?'
        #    fr'([^\.\n]+?\.){{1,{int(task_complexity)}}}?'
        #    r'\n</think>\n'
        #    r'Action:\n\{\n  "name": "[^"\n]+?",\n  "arguments": \["[^\]\n\}]+?"\]\n\}'
        #),
        #final_guide=(
        #    r'I must use the \'final_answer\' tool now\.\n'
        #    fr'([^\.\n]+?\.){{0,{int(task_complexity)}}}?'
        #    r'\n</think>\n'
        #    r'Action:\n\{\n  "name": "final_answer",\n  "arguments": \["[^\]\n\}]+?"\]\n\}'
        #),
        tools=[NoteToSelf()],
        managed_agents=[coder, online_researcher],
        max_steps=max(3, int(task_complexity)),
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

    answer = brain.run(question, additional_args=additional_args, images=images)
    return answer
