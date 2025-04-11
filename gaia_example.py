import concurrent.futures
import os
import threading
from datetime import datetime

import datasets
import mlx.core as mx
import pandas
import transformers
import xgrammar
from rich import print
from tqdm.auto import tqdm

from gaia_scorer import question_scorer
from guided_agents import BaseMLXLogitsProcessor, CodeAgent, MLXLModel, ToolCallingAgent
from guided_agents.default_tools import *
from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class LockedMLXModel:

    def __init__(self, model_id, max_tokens):
        self.lock = threading.Lock()
        self.model = MLXLModel(
            model_id=model_id,
            max_tokens=max_tokens,
            logits_processor=RegexLogitsProcessor
        )
        self.model_id=model_id

    def __call__(self, *args, **kwargs):
        with self.lock:
            return self.model.__call__(*args, **kwargs)


class RegexLogitsProcessor(BaseMLXLogitsProcessor):
    def __init__(self, guide, model_id):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        config = transformers.AutoConfig.from_pretrained(model_id)
        # This can be larger than tokenizer.vocab_size due to paddings
        try:
            full_vocab_size = config.vocab_size
        except:
            full_vocab_size = None
        tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=full_vocab_size)
        grammar_compiler = xgrammar.GrammarCompiler(tokenizer_info)
        compiled_grammar = grammar_compiler.compile_regex(regex=guide)
        self._processor = XGrammarLogitsProcessor(compiled_grammar)

    def __call__(self, input_ids, logits):
        processed_logits = self._processor(
            input_ids,
            logits
        )
        return processed_logits


class XGrammarLogitsProcessor:
    def __init__(self, grammar: xgrammar.CompiledGrammar, max_rollback_tokens: int = 16):
        self.matcher = xgrammar.GrammarMatcher(grammar, max_rollback_tokens=max_rollback_tokens)
        self.vocab_size = grammar.tokenizer_info.vocab_size
        self.bitmask = xgrammar.allocate_token_bitmask(1, self.vocab_size)
        self._prefilled = False

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        last_token = tokens[-1].item()
        if self._prefilled and not self.matcher.is_terminated():
            assert self.matcher.accept_token(last_token)
        else:
            self._prefilled = True
        if not self.matcher.is_terminated():
            self.matcher.fill_next_token_bitmask(self.bitmask)
            return apply_token_bitmask_mlx(
                mx.array(self.bitmask.numpy()), logits, self.vocab_size
            )
        return logits


def run_agents(question, additional_args, log_dir):

    os.makedirs(log_dir, exist_ok=True)

    python_coder = CodeAgent(
        name="python_coder",
        description="Give this agent a task description and it will complete it with Python.",
        model=model,
        guide=(
            r'Code:\n```(?:py|python)?\n[^`]+?\n```<end_code>'
        ),
        final_guide=(
            r'Thought: I must use the \'final_answer\' tool now\.\n'
            r'Code:\n```(?:py|python)?\n[^`]+?\n```<end_code>'
        ),
        tools=[DuckDuckGoSearchTool()],
        max_steps=5,
        verbosity_level=1,
        log_dir=log_dir,
    )
    planner = ToolCallingAgent(
        name="planner",
        model=model,
        guide=(
            r'Thought: [^\n\.]+?\.\n'
            r'Action:\n\{\n  "name": "[^"\n]+?",\n  "arguments": "[^\n\}]+?"\n\}'
        ),
        final_guide=(
            r'Thought: I must use the \'final_answer\' tool now\.\n'
            r'Action:\n\{\n  "name": "final_answer",\n  "arguments": "[^\n\}]+?"\n\}'
        ),
        managed_agents=[
            python_coder,
        ],
        tools=[],
        max_steps=5,
        verbosity_level=1,
        log_dir=log_dir,
    )

    final_answer_requirements = (
        "\nYour final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings."
        " If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign unless specified otherwise."
        " If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise."
        " If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
    )
    tool_prompt = (
        "\nYou only have access to these tools:\n\n{%- for tool in tools.values() %}\n- {{ tool.name }}: {{ tool.description }}"
        "\n      Args:{%- for k,v in tool.inputs.items() %}"
        "\n        {{k}}: {{v['description']}}"
        "\n{%- endfor %}"
        "\n      Returns: {{tool.output_type}}"
        "{%- endfor %}"
    )
    manager_prompt = (
        "{%- if managed_agents and managed_agents.values() | list %}"
        "\nYou can also call agents to give you answers to simple tasks."
        " To call an agent, print a newline and say:\nAction:\n{\n  \"name\": \"...\"\n  \"arguments\": \"..."
        " Here is a list of agents that you can call:"
        "{%- for agent in managed_agents.values() %}"
        "\n- {{ agent.name }}: {{ agent.description }}"
        "{%- endfor %}"
        "{%- else %}"
        "{%- endif %}"
    )
    system_prompt = (
        "You execute tasks given to you as correctly and efficiently as possible. "
        "Take small steps as you learn new information. "
    ) + tool_prompt + manager_prompt

    planner.prompt_templates["system_prompt"] = system_prompt + final_answer_requirements

    python_coder.prompt_templates["system_prompt"] = system_prompt + "\nCall tools from your Python console. The tools are already imported for you. To use your Python console, print a newline and say:\nCode:\n```python\n"
    python_coder.prompt_templates["managed_agent"]["task"] = "{{task}}"
    python_coder.prompt_templates["managed_agent"]["report"] = "{{final_answer}}"

    answer = planner.run(question, additional_args=additional_args)
    return answer


def run(question, additional_args, log_dir):
    try:
        answer = run_agents(question, additional_args, log_dir)
        return answer
    except Exception as e:
        msg = f"ERROR: {e}"
        print(msg)
        return msg


gaia_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_level1")
gaia_ds_val = gaia_ds["validation"]

model = LockedMLXModel(
    model_id="mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
    max_tokens=1000,
)
#model = brain_model = OpenAIServerModel(
#    model_id="stelterlab/Mistral-Small-24B-Instruct-2501-AWQ",
#    api_base="http://192.168.1.69:8000/v1",
#    api_key=os.environ["VLLM_API_KEY"],
#    max_tokens=5000,
#    temperature=0.0,
#)

questions = []
file_paths = []
correct_answers = []

for task in gaia_ds_val:
    questions.append(task["Question"])
    file_paths.append(task["file_path"])
    correct_answers.append(task["Final answer"])

progress_bar = tqdm(total=len(questions), desc="Tasks Complete", unit="task", bar_format="{l_bar}{bar}| {rate_fmt} {n_fmt}/{total_fmt}{postfix}")
correct_answers_count = 0
base_log_dir = f"./logs/{datetime.now():%Y-%m-%d_%H:%M:%S%z}"

# Run multiple tasks at once - useful for OpenAIServerModel type
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    futures = {}
    for i,question in enumerate(questions):
        additional_args = {}
        if file_paths[i]:
            additional_args["file_path"] = file_paths[i]
        log_dir = f"{base_log_dir}/{i+1:02}"
        futures[executor.submit(run, question, additional_args, log_dir)] = i
    answers = {}
    correct = {}
    for future in concurrent.futures.as_completed(futures):
        progress_bar.update(1)
        i = futures[future]
        answers[i] = future.result()
        correct[i] = question_scorer(str(answers[i]), correct_answers[i])
        correct_answers_count += int(correct[i])
        progress_bar.set_postfix_str(f"Accuracy: {correct_answers_count / len(correct)}")

scores = pandas.DataFrame(
    {
        "correct": [correct[k] for k in sorted(correct)],
        "answer": [answers[k] for k in sorted(answers)],
        "correct_answer": correct_answers,
        "question": questions,
    }
)
scores.index += 1
scores.to_csv(base_log_dir + "/scores.csv", index_label="task_id")

val_submission = pandas.DataFrame(
    {
        "task_id": [f"task_id_{i}" for i in scores.index],
        "model_answer": scores["answer"].values,
    }
)
val_submission.to_csv(base_log_dir + "/val_submission.csv", index=False)

print(scores)
