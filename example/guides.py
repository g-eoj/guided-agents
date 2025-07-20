from typing import List


class LarkReasoningGuide:
    def __init__(
        self,
        act: str|None=None,
        model_id: str|None=None,
        reasoning_paragraph_limit: int=3,
        reasoning_sentence_limit: int=5,
        think_start_token_ids: List[int]|None=None,
        think_stop_token_ids: List[int]|None=None,
    ):
        self.act = act or "act: /.+/\n"  # default act should allow any tokens
        self.reasoning_paragraph_limit = reasoning_paragraph_limit
        self.reasoning_sentence_limit = reasoning_sentence_limit
        self.think_start_token_ids = None
        self.think_stop_token_ids = None

        # get token IDs from reasoning model tokenizer
        if model_id:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.think_start_token_ids = tokenizer.encode("<think>")
            self.think_stop_token_ids = tokenizer.encode("</think>")

        # optionally override token IDs instead of using tokenizer
        if think_start_token_ids is not None:
            self.think_start_token_ids = think_start_token_ids
        if think_stop_token_ids is not None:
            self.think_stop_token_ids = think_stop_token_ids

        # construct guide if possible
        if self.think_start_token_ids and self.think_stop_token_ids:
            self.guide = (
                f'start: <{self.think_start_token_ids}> reason <{self.think_stop_token_ids}> NL act\n'
                f'reason: paragraph{{1,{int(self.reasoning_paragraph_limit)}}}\n'
                f'paragraph: NL sentence{{1,{int(self.reasoning_sentence_limit)}}} NL\n'
                f'sentence[lazy]: /[^\\.\\n]+/ (".")\n'
                f'{self.act}'
                'NL: /\\n/\n'
                'Q: /"/\n'
            )
        else:
            raise RuntimeError("Token IDs for reasoning not found or not given.")

    def __call__(self):
        return self.guide


class RegexReasoningGuide:
    def __init__(
        self,
        act: str|None=None,
        reasoning_sentence_limit: int=5,
        think_start_str: str="Thought: ",
        think_stop_str: str="\n",
    ):
        self.act = act or ".+"  # default act should allow any tokens
        self.reasoning_sentence_limit = reasoning_sentence_limit
        self.think_start_str = think_start_str
        self.think_stop_str = think_stop_str
        self.guide = self.think_start_str
        self.guide += fr'([^\n\.]+?\.){{1,{int(self.reasoning_sentence_limit)}}}\n\n'
        self.guide += self.think_stop_str
        self.guide += self.act

    def __call__(self):
        return self.guide
