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

from abc import abstractmethod
from guided_agents import Tool


class GoogleScholarSearchTool(Tool):
    name = "paper_search"
    description = "Search the web for research papers to read."
    inputs = {
        "query": {"type": "string", "description": "The topic or title of a paper you are looking for. Don't give this tool general queries as it is not designed to return comprehensive results."},
    }
    output_type = "string"

    def __init__(self, api_key):
        super().__init__(self)
        import requests_cache
        self.api_key = api_key
        self.headers = {'X-API-KEY': api_key}
        self.url = "https://google.serper.dev/scholar"
        self.session = requests_cache.CachedSession("google_search_cache", allowable_methods=["POST"])

    def forward(self, query: str) -> str:
        result = self.session.post(url=self.url, data={"q": query}, headers=self.headers)
        result.raise_for_status()
        result = result.json()["organic"]
        text = "\n"
        for r in result:
            pdf_url = r.get("pdfUrl", None)
            html_url = r.get("htmlUrl", None)
            link_url = r.get("link", None)
            link = pdf_url or html_url or link_url
            text += f"- [{r['title']}]({link}): {r.get('snippet', '')}...\n\tPublication Information: {r.get('publicationInfo', '')}\n"
        return text


class GoogleSearchTool(Tool):
    name = "web_search"
    description = "Search the web for links."
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
    }
    output_type = "string"

    def __init__(self, api_key):
        super().__init__(self)
        import requests_cache
        self.api_key = api_key
        self.headers = {'X-API-KEY': api_key}
        self.url = "https://google.serper.dev/search"
        self.session = requests_cache.CachedSession("google_search_cache", allowable_methods=["POST"])

    def forward(self, query: str) -> str:
        result = self.session.post(url=self.url, data={"q": query}, headers=self.headers)
        result.raise_for_status()
        result = result.json()["organic"]
        text = "\n"
        for r in result:
            text += f"- [{r['title']}]({r['link']})\n\tSnippet: {r.get('snippet', '')}\n"
        return text


class NoteToSelf(Tool):
    name = "note_to_self"
    description = "Leave a note about anything you want to remember."
    inputs = {
        "note": {"type": "string", "description": "The note."},
    }
    output_type = "string"

    def forward(self, note: str) -> str:
        import pprint
        return pprint.pformat(note)


class Reader(Tool):

    def __init__(
        self, model, guide=None, max_iterations_per_page=100, max_workers=1, min_notes_if_possible=3, logger=None, path=None
    ):
        super().__init__(self)
        self.model = model
        self.logger = logger
        self.path = path
        self.guide = guide
        self.max_iterations_per_page = max_iterations_per_page
        self.max_workers = max_workers
        self.min_notes_if_possible = min_notes_if_possible

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model.model_id)
        self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            separators=["\n#+", "\n\n", "\n", " ", ""],
            is_separator_regex=True,
            tokenizer=tokenizer,
            chunk_overlap=6000,
            chunk_size=9000
        )

    def generate(guide, messages, model):
        return model(
            messages=messages,
            guide=guide,
        ).content

    @abstractmethod
    def get_md(self, path: str):
        ...

    def forward(self, path: str, query: str):

        try:
            if self.path:
                path = self.path
            md = self.get_md(path)
        except ValueError as e:
            print(e)
            return str(e)

        messages_per_chunk = []
        system_text = (
            "You carefully find and organize information. "
            "When you find relevant information, you make a note like this: 'Note: ...'"
        )
        for chunk in self.text_splitter.split_text(md):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content":
                    [
                        {
                            "type": "text",
                            "text": (
                                f"Look for information relevant to '{query}' in this chunk of markdown:\n\n{chunk}\n\n"
                                "Now, only using the chunk of markdown, make a note of any relevant information. "
                                "If you don't find any relevant informtation, then don't make a note. "
                                "Just say '<not_relevant>'."
                            )
                        }
                    ]
                }
            ]
            messages_per_chunk.append(messages)

        notes = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        futures = [executor.submit(WebReader.generate, self.guide, messages, self.model) for messages in messages_per_chunk]
        for future in futures:
            if future.cancelled():
                continue
            note = future.result()
            if self.logger:
                model_name = f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}"
                self.logger.info(msg=note, extra={"agent": self.name, "model": model_name, "stage": "RESEARCH"})
            note = note.split('</think>')[-1].strip()
            if not note.startswith("<not_relevant>"):
                notes.append(note)
            if len(notes) >= self.min_notes_if_possible:
                executor.shutdown(wait=True, cancel_futures=True)

        if len(notes) > 1:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content":
                    [
                        {
                            "type": "text",
                            "text": (
                                f"Combine these notes to help answer '{query}'. "
                                "Check if the notes have contradictory information. "
                                f"If they do, explain the contradictions:\n\n{notes}"
                            )
                        }
                    ]
                }
            ]
            answer = Reader.generate(self.guide, messages, self.model)
            if self.logger:
                model_name = f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}"
                self.logger.info(msg=answer, extra={"agent": self.name, "model": model_name, "stage": "RESEARCH"})
            answer = answer.split('</think>')[-1].lstrip('\\n')
            return answer
        elif notes:
            return notes[0]
        else:
            return "No answers found. Please use a different link or rephrase your query."


class FileReader(Reader):
    name = "file_reader"
    description = "Answers your query about file contents."
    inputs = {
        "path": {
            "type": "string",
            "description": "The file to use."
        },
        "query": {
            "type": "string",
            "description": "Ask a simple but detailed question about what you want to learn."
        }
    }
    output_type = "string"

    def get_md(self, path:str):
        import pathlib
        from file_loaders import DOCXReader, MLXAudioTranscribe, PPTXReader, ExcelReader, TXTReader
        file_loader = None
        file_name = pathlib.Path(path).name
        file_suffix = pathlib.Path(path).suffix
        if file_suffix in [".docx"]:
            file_loader = DOCXReader(path)
        if file_suffix in [".mp3"]:
            file_loader = MLXAudioTranscribe(path)
        if file_suffix in [".pptx"]:
            file_loader = PPTXReader(path)
        if file_suffix in [".xlsx"]:
            file_loader = ExcelReader(path)
        if file_suffix in [".txt", ".py"]:
            file_loader = TXTReader(path)
        if file_loader:
            return file_loader.forward()
        return ""


class WebReader(Reader):
    name = "web_reader"
    description = "Converts a web page or PDF link to markdown, then answers your query about the markdown content."
    inputs = {
        "path": {
            "type": "string",
            "description": "The URL or file path to read from."
        },
        "query": {
            "type": "string",
            "description": "Ask a simple but detailed question about what you want to learn."
        }
    }
    output_type = "string"

    def get_md(self, path:str):
        import re
        import time
        from urllib.parse import urlparse

        import markdownify
        from playwright.sync_api import sync_playwright

        hostname = urlparse(path).hostname
        if hostname in ["www.huggingface.co", "www.researchgate.net"]:
            raise ValueError(f"{hostname} is forbidden. Try a different website.")
        md = ""
        with sync_playwright() as playwright:
            browser = playwright.firefox.launch(
                firefox_user_prefs={
                    "pdfjs.disabled": False,
                    "browser.download.open_pdf_attachments_inline": True,
                    "browser.link.open_newwindow": 1,
                }
            )
            page = browser.new_page()
            page.goto(path, wait_until="commit")
            time.sleep(3)
            for frame in page.frames:
                try:
                    # try loading the pdf viewer
                    content = frame.inner_html("id=viewer", timeout=1000)
                except Exception:
                    content = frame.page.inner_html("body")
                md += markdownify.markdownify(
                    content,
                    strip=["a"],
                    heading_style="ATX",
                    table_infer_header=True,
                ) + "\n\n"
            browser.close()
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md
