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

import pdf2image
from PIL import Image, ImageChops

from guided_agents import Tool


class DOCXReader(Tool):
    name = "docx_reader"
    description = "This is a tool that converts docx files into markdown."
    inputs = {}
    output_type = "string"
    skip_forward_signature_validation = True

    def __init__(self, file_path):
        super().__init__(self)
        self.file_path = file_path

    def forward(self, *args):
        import re

        import docx
        from docx import Document
        text = ""

        document = Document(self.file_path)
        for x in document.iter_inner_content():
            if isinstance(x, docx.text.paragraph.Paragraph):
                if "heading" in x.style.name.lower():
                    text += "#" * int(x.style.name.lower().split()[-1]) + " "
                if "list" in x.style.name.lower():
                    text += "- "
                text += x.text + "\n"
            elif isinstance(x, docx.table.Table):
                for row in x.rows:
                    for cell in row.cells:
                        text += cell.text + ", "
                    text += "\n"
        text = re.sub(r'[\n]{3,}', "\n\n", text)
        return text


class ExcelReader(Tool):
    name = "excel_reader"
    description = (
        "This is a tool that converts excel files into text. "
        "Each sheet will be followed by two tables of CSVs. "
        "The first table contains the values of the cells in the sheet. "
        "The second table contains the style ID of the cell."
    )
    inputs = {}
    output_type = "string"
    skip_forward_signature_validation = True

    def __init__(self, file_path):
        super().__init__(self)
        self.file_path = file_path

    def forward(self, *args):
        import openpyxl

        work_book = openpyxl.open(self.file_path)
        text = ""
        for sheet in work_book:
            values = ""
            style_ids = ""
            text += str(sheet) + "\n"
            for row in sheet.iter_rows():
                for cell in row:
                    values += str(cell.value) + ", "
                    style_ids += str(cell.style_id) + ", "
                values = values[:-2] + "\n"
                style_ids = style_ids[:-2] + "\n"
            text += values
            text += "\n"
            #text += style_ids
            #text += "\n"
        text +="\n"
        return text


class MLXAudioTranscribe(Tool):
    name = "audio_transcribe"
    description = "Transcribe speech from audio files."
    inputs = {}
    output_type = "string"
    skip_forward_signature_validation = True

    def __init__(self, file_path):
        super().__init__(self)
        self.file_path = file_path

    def forward(self) -> str:
        import mlx_whisper
        text = mlx_whisper.transcribe(self.file_path)["text"]
        return text


class PPTXReader(Tool):
    name = "pptx_reader"
    description = "This tool converts pptx files into a list. Each element of the list is text from one slide."
    inputs = {}
    output_type = "array"
    skip_forward_signature_validation = True

    def __init__(self, file_path):
        super().__init__(self)
        self.file_path = file_path

    def forward(self, *args) -> str:
        from pptx import Presentation
        presentation = Presentation(self.file_path)
        slides_text = []
        for slide in presentation.slides:
            text = ""
            element = slide.element
            for x in element.iterdescendants():
                if x.text and hasattr(x, "xml") and x.xml.startswith("<a:r"):
                    text += x.text + "\n"
            slides_text.append(text)
        return slides_text


class TXTReader(Tool):
    name = "txt_reader"
    description = "This is a tool that converts txt files into text."
    inputs = {}
    output_type = "string"
    skip_forward_signature_validation = True

    def __init__(self, file_path):
        super().__init__(self)
        self.file_path = file_path

    def forward(self, *args):
        with open(self.file_path, "r") as h:
            text = "".join(h.readlines())
        return text


# helper functions
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
