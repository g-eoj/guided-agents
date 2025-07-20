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


import os
import threading

import mlx.core as mx
import transformers
import xgrammar
from xgrammar.kernels.apply_token_bitmask_mlx import apply_token_bitmask_mlx

from guided_agents import BaseMLXLogitsProcessor, MLXLModel, OpenAIServerModel


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


class GrammarLogitsProcessor(BaseMLXLogitsProcessor):
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
        compiled_grammar = grammar_compiler.compile_grammar(guide)
        self._processor = XGrammarLogitsProcessor(compiled_grammar)

    def __call__(self, input_ids, logits):
        processed_logits = self._processor(
            input_ids,
            logits
        )
        return processed_logits


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

# mistralrs
#coder_model = OpenAIServerModel(
#    model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
#    api_base="http://0.0.0.0:1234/v1",
#    api_key="abc",
#    max_tokens=2000,
#    temperature=0.0,
#)

# MLX-LM model
#coder_model = LockedMLXModel(
#    model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
#    max_tokens=3000,
#)

#coder_model = model = MLXLModel(
#    model_id="mlx-community/Qwen2.5-7B-Instruct-4bit",
#    model_id="mlx-community/Mistral-Small-24B-Instruct-2501-4bit",
#    model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
#    model_id="mlx-community/QwQ-32B-Preview-6bit",
#    model_id="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
#    model_id="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
#    logits_processor=GrammarLogitsProcessor,
#    max_tokens=5000,
#)

# MLX-VLM model
#model = MLXVLModel(
#    model_id="mlx-community/Mistral-Small-3.1-24B-Instruct-2503-4bit",
#    logits_processor=RegexLogitsProcessor,
#    max_tokens=5000,
#)

# Local vLLM model
model = OpenAIServerModel(
    model_id="Qwen/Qwen3-8B",
    api_base="http://10.0.0.33:8000/v1",
    api_key=os.environ["VLLM_API_KEY"],
    max_tokens=3000,
    temperature=0.02,
)

coder_model = OpenAIServerModel(
    model_id="Qwen/Qwen3-8B",
    api_base="http://10.0.0.33:8000/v1",
    api_key=os.environ["VLLM_API_KEY"],
    max_tokens=3000,
    temperature=0.01,
)

# RunPod vLLM model
#pod_id = "pjpkjwyp8nsc17"
#model = OpenAIServerModel(
#    model_id="Qwen/Qwen3-32B",
#    api_base=f"https://{pod_id}-8000.proxy.runpod.net/v1",
#    api_key=os.environ["VLLM_API_KEY"],
#    max_tokens=3000,
#    #temperature=0.0,
#)
#
#coder_model = OpenAIServerModel(
#    model_id="Qwen/Qwen3-32B",
#    api_base=f"https://{pod_id}-8000.proxy.runpod.net/v1",
#    api_key=os.environ["VLLM_API_KEY"],
#    max_tokens=3000,
#    temperature=0.0,
#)
