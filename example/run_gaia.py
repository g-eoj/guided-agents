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


import concurrent.futures
import os
import traceback
from datetime import datetime


os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import datasets
import pandas
from core import run_agents
from gaia.scorer import question_scorer
from rich import print
from tqdm.auto import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(question, file_path, log_dir, first_to=3, max_samples=5):
    answer_counts = {}
    try:
        for i in range(max_samples):
            answer = run_agents(question, file_path, log_dir=f"{log_dir}-{i+1}", strict_answers=True)
            if answer in answer_counts:
                answer_counts[answer] += 1
            else:
                answer_counts[answer] = 1
            if answer_counts[answer] == first_to:
                return answer
        return None
    except Exception as e:
        traceback.print_exc()
        msg = f"ERROR: {e}"
        print(msg)
        return msg


# configure run
gaia_ds = datasets.load_dataset("gaia-benchmark/GAIA", "2023_level1")
gaia_ds_val = gaia_ds["validation"]
first_to = 3
max_workers = 2
max_samples = 5

task_ids = []
questions = []
file_paths = []
correct_answers = []
counter = 0
for task in gaia_ds_val:
    #counter += 1
    #if counter != 1:
    #    continue
    task_ids.append(task["task_id"])
    questions.append(task["Question"])
    file_paths.append(task["file_path"])
    correct_answers.append(task["Final answer"])

progress_bar = tqdm(total=len(questions), desc="Tasks Complete", unit="task", bar_format="{l_bar}{bar}| {rate_fmt} {n_fmt}/{total_fmt}{postfix}")
correct_answers_count = 0

base_log_dir = f"./logs/{datetime.now():%Y-%m-%d_%H:%M:%S%z}"
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {}
    for i,question in enumerate(questions):
        log_dir = f"{base_log_dir}/{i+1:02}"
        futures[executor.submit(run, question, file_paths[i], log_dir, first_to, max_samples)] = i
    answers = {}
    correct = {}
    for future in concurrent.futures.as_completed(futures):
        i = futures[future]
        answers[i] = future.result()
        correct[i] = question_scorer(
            str(answers[i]),
            correct_answers[i]
        )
        correct_answers_count += int(correct[i])
        progress_bar.update(1)
        progress_bar.set_postfix_str(f"Accuracy: {round(correct_answers_count / len(correct), 2)}")

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
        "task_id": task_ids,
        "model_answer": scores["answer"].values,
    }
)
val_submission.to_csv(base_log_dir + "/val_submission.csv", index=False)

print(scores)
