from pathlib import Path
from html2text import html2text
from langchain.text_splitter import TokenTextSplitter
from typing import Any, Callable, Dict, List, Optional, Set, Union
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, validator
from prompts_config import citation_prompt, qa_prompt, select_paper_prompt, summary_prompt

Doc_Key = Any
StrPath = Union[str, Path]
CBManager = Union[AsyncCallbackManagerForChainRun, CallbackManagerForChainRun]
CallbackFactory = Callable[[str], Union[None, List[BaseCallbackHandler]]]


class Doc(BaseModel):
    doc_name: str
    citation: str
    doc_key: Doc_Key


class Text(BaseModel):
    text: str
    name: str
    doc: Doc
    embeddings: Optional[List[float]] = None

class PromptCollection(BaseModel):
    summary: PromptTemplate = summary_prompt
    qa: PromptTemplate = qa_prompt
    select: PromptTemplate = select_paper_prompt
    cite: PromptTemplate = citation_prompt
    pre: Optional[PromptTemplate] = None
    post: Optional[PromptTemplate] = None

    @validator("summary")
    def check_summary(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(summary_prompt.input_variables)):
            raise ValueError(
                f"Summary prompt can only have variables: {summary_prompt.input_variables}"
            )
        return v

    @validator("qa")
    def check_qa(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(set(qa_prompt.input_variables)):
            raise ValueError(
                f"QA prompt can only have variables: {qa_prompt.input_variables}"
            )
        return v

    @validator("select")
    def check_select(cls, v: PromptTemplate) -> PromptTemplate:
        if not set(v.input_variables).issubset(
            set(select_paper_prompt.input_variables)
        ):
            raise ValueError(
                f"Select prompt can only have variables: {select_paper_prompt.input_variables}"
            )
        return v

    @validator("pre")
    def check_pre(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            if set(v.input_variables) != set(["question"]):
                raise ValueError("Pre prompt must have input variables: question")
        return v

    @validator("post")
    def check_post(cls, v: Optional[PromptTemplate]) -> Optional[PromptTemplate]:
        if v is not None:
            # kind of a hack to get list of attributes in answer
            attrs = [a.name for a in Answer.__fields__.values()]
            if not set(v.input_variables).issubset(attrs):
                raise ValueError(f"Post prompt must have input variables: {attrs}")
        return v
        
        
class Context(BaseModel):
    """A class to hold the context of a question."""

    context: str
    text: Text
    score: int = 5

        
def __str__(self) -> str:
    """Return the context as a string."""
    return self.context
        
        
        
class Answer(BaseModel):
    """A class to hold the answer to a question."""

    question: str
    answer: str = ""
    context: str = ""
    contexts: List[Context] = []
    references: str = ""
    formatted_answer: str = ""
    doc_key_filter: Optional[Set[Doc_Key]] = None
    summary_length: str = "about 100 words"
    answer_length: str = "about 100 words"
    memory: Optional[str] = None
    # these two below are for convenience
    # and are not set. But you can set them
    # if you want to use them.
    cost: Optional[float] = None
    token_counts: Optional[Dict[str, List[int]]] = None

    def __str__(self) -> str:
        """Return the answer as a string."""
        return self.formatted_answer
    
    
def parse_pdf_fitz(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    import fitz

    file = fitz.open(path)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i in range(file.page_count):
        page = file.load_page(i)
        split += page.get_text("text", sort=True)
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.doc_name} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.doc_name} pages {pg}", doc=doc)
        )
    file.close()
    return texts


def parse_pdf(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    import pypdf

    pdfFileObj = open(path, "rb")
    pdfReader = pypdf.PdfReader(pdfFileObj)
    split = ""
    pages: List[str] = []
    texts: List[Text] = []
    for i, page in enumerate(pdfReader.pages):
        split += page.extract_text()
        pages.append(str(i + 1))
        # split could be so long it needs to be split
        # into multiple chunks. Or it could be so short
        # that it needs to be combined with the next chunk.
        while len(split) > chunk_chars:
            # pretty formatting of pages (e.g. 1-3, 4, 5-7)
            pg = "-".join([pages[0], pages[-1]])
            texts.append(
                Text(
                    text=split[:chunk_chars], name=f"{doc.doc_name} pages {pg}", doc=doc
                )
            )
            split = split[chunk_chars - overlap :]
            pages = [str(i + 1)]
    if len(split) > overlap:
        pg = "-".join([pages[0], pages[-1]])
        texts.append(
            Text(text=split[:chunk_chars], name=f"{doc.doc_name} pages {pg}", doc=doc)
        )
    pdfFileObj.close()
    return texts


def parse_txt(
    path: Path, doc: Doc, chunk_chars: int, overlap: int, html: bool = False
) -> List[Text]:
    try:
        with open(path) as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    if html:
        text = html2text(text)
    # yo, no idea why but the texts are not split correctly
    text_splitter = TokenTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap)
    raw_texts = text_splitter.split_text(text)
    texts = [
        Text(text=t, name=f"{doc.doc_name} chunk {i}", doc=doc)
        for i, t in enumerate(raw_texts)
    ]
    return texts


def parse_code_txt(path: Path, doc: Doc, chunk_chars: int, overlap: int) -> List[Text]:
    """Parse a document into chunks, based on line numbers (for code)."""

    split = ""
    texts: List[Text] = []
    last_line = 0

    with open(path) as f:
        for i, line in enumerate(f):
            split += line
            if len(split) > chunk_chars:
                texts.append(
                    Text(
                        text=split[:chunk_chars],
                        name=f"{doc.doc_name} lines {last_line}-{i}",
                        doc=doc,
                    )
                )
                split = split[chunk_chars - overlap :]
                last_line = i
    if len(split) > overlap:
        texts.append(
            Text(
                text=split[:chunk_chars],
                name=f"{doc.doc_name} lines {last_line}-{i}",
                doc=doc,
            )
        )
    return texts


def read_doc(
    path: Path,
    doc: Doc,
    chunk_chars: int = 3000,
    overlap: int = 100,
    force_pypdf: bool = False,
) -> List[Text]:
    """Parse a document into chunks."""
    str_path = str(path)
    if str_path.endswith(".pdf"):
        if force_pypdf:
            return parse_pdf(path, doc, chunk_chars, overlap)
        try:
            return parse_pdf_fitz(path, doc, chunk_chars, overlap)
        except ImportError:
            return parse_pdf(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".txt"):
        return parse_txt(path, doc, chunk_chars, overlap)
    elif str_path.endswith(".html"):
        return parse_txt(path, doc, chunk_chars, overlap, html=True)
    else:
        return parse_code_txt(path, doc, chunk_chars, overlap)
