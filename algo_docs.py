import asyncio
import os
import re
import sys
import tempfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Set, Union, cast

from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.base import Embeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationTokenBufferMemory
from langchain.memory.chat_memory import BaseChatMemory

from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import FAISS
from pydantic import BaseModel, validator

from doc_chains import get_score, make_chain
from prompts_config import BASE_DIR
from doc_parser import read_doc
from doc_parser import Answer, CallbackFactory, Context, Doc, Doc_Key, PromptCollection, Text
from utils import (
    gather_with_concurrency,
    maybe_is_html,
    maybe_is_pdf,
    maybe_is_text,
    md5sum,
    name_in_text,
)


class AlgoDocs():
    
    def __init__(self, llm = ChatOpenAI(temperature=0.1, model="gpt-4", client=None), summary_llm = ChatOpenAI(temperature=0.1, model="gpt-4", client=None)):

        self.docs = {}
        self.texts = []
        self.doc_names = set()
        self.texts_index = None
        self.doc_index = None

        self.llm = llm
        self.summary_llm = summary_llm

        self.name = "default"
        self.index_path = BASE_DIR / self.name

        self.embeddings = OpenAIEmbeddings(client=None, chunk_size=1000, 
            deployment='text-embedding-ada-002', disallowed_special='all', embedding_ctx_length=8191,
            headers=None, model='text-embedding-ada-002', )


        self.max_concurrent = 5
        self.prompts = PromptCollection()
    
    def _get_unique_name(self, doc_name: str) -> str:
        """Create a unique name given proposed name"""
        suffix = ""
        while doc_name + suffix in self.doc_names:
            # move suffix to next letter
            if suffix == "":
                suffix = "a"
            else:
                suffix = chr(ord(suffix) + 1)
        doc_name += suffix
        return doc_name

    def get_context_and_summary(self, answer):
        return answer.answer, answer.context


    def add(self, path, citation = None, doc_name = None, disable_check = False, doc_key = None, chunk_chars = 3000):
        """Add a document to the collection."""
        
        if doc_key is None:
            doc_key = md5sum(path)
        if citation is None:
            cite_chain = make_chain(
                prompt=self.prompts.cite,
                llm=cast(BaseLanguageModel, self.summary_llm),
                skip_system=True,
            )
            # peak first chunk
            fake_doc = Doc(doc_name="", citation="", doc_key=doc_key)
            texts = read_doc(path, fake_doc, chunk_chars=chunk_chars, overlap=300)
            if len(texts) == 0:
                raise ValueError(f"Could not read document {path}. Is it empty?")
            citation = cite_chain.run(texts[0].text)
            if len(citation) < 3 or "Unknown" in citation or "insufficient" in citation:
                citation = f"Unknown, {os.path.basename(path)}, {datetime.now().year}"

        if doc_name is None:
            # get first name and year from citation
            match = re.search(r"([A-Z][a-z]+)", citation)
            if match is not None:
                author = match.group(1)  # type: ignore
            else:
                # panicking - no word??
                raise ValueError(
                    f"Could not parse doc_name from citation {citation}. "
                    "Consider just passing key explicitly - e.g. docs.py "
                    "(path, citation, key='mykey')"
                )
            year = ""
            match = re.search(r"(\d{4})", citation)
            if match is not None:
                year = match.group(1)  # type: ignore
            doc_name = f"{author}{year}"
        doc_name = self._get_unique_name(doc_name)
        doc = Doc(doc_name=doc_name, citation=citation, doc_key=doc_key)
        texts = read_doc(path, doc, chunk_chars=chunk_chars, overlap=100)
        # loose check to see if document was loaded
        if (
            len(texts) == 0
            or len(texts[0].text) < 10
            or (not disable_check and not maybe_is_text(texts[0].text))
        ):
            raise ValueError(
                f"This does not look like a text document: {path}. Path disable_check to ignore this error."
            )
        if self.add_texts(texts, doc):
            return doc_name
        return None

    def add_texts(self, texts, doc):
        """Add chunked texts to the collection.
        Returns True if the document was added, False if it was already in the collection.
        """
        if doc.doc_key in self.docs:
            return False
        if len(texts) == 0:
            raise ValueError("No texts to add.")
        if doc.doc_name in self.doc_names:
            new_doc_name = self._get_unique_name(doc.doc_name)
            for t in texts:
                t.name = t.name.replace(doc.doc_name, new_doc_name)
            doc.doc_name = new_doc_name
        if texts[0].embeddings is None:
            text_embeddings = self.embeddings.embed_documents([t.text for t in texts])
            for i, t in enumerate(texts):
                t.embeddings = text_embeddings[i]
        else:
            text_embeddings = cast(List[List[float]], [t.embeddings for t in texts])
        if self.texts_index is not None:
            try:
                # TODO: Simplify - super weird
                vec_store_text_and_embeddings = list(
                    map(lambda x: (x.text, x.embeddings), texts)
                )
                self.texts_index.add_embeddings(  # type: ignore
                    vec_store_text_and_embeddings,
                    metadatas=[t.dict(exclude={"embeddings", "text"}) for t in texts],
                )
            except AttributeError:
                raise ValueError("Need a vector store that supports adding embeddings.")
        if self.doc_index is not None:
            self.doc_index.add_texts([doc.citation], metadatas=[doc.dict()])
        self.docs[doc.doc_key] = doc
        self.texts += texts
        self.doc_names.add(doc.doc_name)
        return True

    def _build_texts_index(self):
        if self.texts_index is None:
            raw_texts = [t.text for t in self.texts]
            text_embeddings = [t.embeddings for t in self.texts]
            metadatas = [t.dict(exclude={"embeddings", "text"}) for t in self.texts]
            self.texts_index = FAISS.from_embeddings(
                text_embeddings=list(zip(raw_texts, text_embeddings)), embedding=self.embeddings, 
                metadatas=metadatas,)

    async def aget_evidence(self, answer, k = 3, max_sources = 5, marginal_relevance = True):
        
        if len(self.docs) == 0 and self.doc_index is None:
            return answer
        
        if self.texts_index is None:
            self._build_texts_index()
        
        self.texts_index = cast(VectorStore, self.texts_index)
        
        _k = k
        if answer.doc_key_filter is not None:
            _k = k * 10
        
        if marginal_relevance:
            matches = self.texts_index.max_marginal_relevance_search(answer.question, k=_k, fetch_k=5 * _k)
        else:
            matches = self.texts_index.similarity_search(answer.question, k=_k, fetch_k=5 * _k)
        
        if answer.doc_key_filter is not None:
            matches = [m for m in matches if m.metadata["doc"]["doc_key"] in answer.doc_key_filter]

        # check if it is already in answer
        cur_names = [c.text.name for c in answer.contexts]
        matches = [m for m in matches if m.metadata["name"] not in cur_names]
        matches = matches[:k]

        async def process(match):
            summary_chain = make_chain(self.prompts.summary, self.summary_llm)
            try:
                context = await summary_chain.arun(question=answer.question, citation=match.metadata["doc"]["citation"],
                    summary_length=answer.summary_length, text=match.page_content,)
            except Exception as e:
                if re.search(r"4\d\d", str(e)):
                    return None
                raise e
            
            if "not applicable" in context.lower():
                return None
            
            return Context(context=context, text=Text(text=match.page_content, name=match.metadata["name"], 
                                                   doc=Doc(**match.metadata["doc"]),),score=get_score(context),)

        results = await gather_with_concurrency(
            self.max_concurrent, *[process(m) for m in matches]
        )
        # filter out failures
        contexts = [c for c in results if c is not None]
        if len(contexts) == 0:
            return answer
        contexts = sorted(contexts, key=lambda x: x.score, reverse=True)
        contexts = contexts[:max_sources]
        # add to answer contexts
        answer.contexts += contexts
        context_str = "\n\n".join(
            [f"{c.text.name}: {c.context}" for c in answer.contexts]
        )
        valid_names = [c.text.name for c in answer.contexts]
        context_str += "\n\nValid keys: " + ", ".join(valid_names)
        answer.context = context_str
        return answer

    def query(self, query, k = 10, max_sources = 5, length_prompt = "about 100 words", marginal_relevance = True, answer = None):
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.aquery(query, k=k, max_sources=max_sources, length_prompt=length_prompt, 
                                                   marginal_relevance=marginal_relevance, answer=answer,))

    def query_get_summary_context(self, query, k = 10, max_sources = 5, length_prompt = "about 100 words", marginal_relevance = True, answer = None):
        answer = self.query(query, k = 10, max_sources = 5, length_prompt = "about 100 words", marginal_relevance = True, answer = None)
        answer, context = self.get_context_and_summary(answer)
        return answer, context


    async def aquery(self, query, k = 10, max_sources = 5, length_prompt = "about 100 words", marginal_relevance = True, 
                     answer = None):
    
        if k < max_sources:
            raise ValueError("k should be greater than max_sources")
            
        if answer is None:
            answer = Answer(question=query, answer_length=length_prompt)
        
        if len(answer.contexts) == 0:
            answer = await self.aget_evidence(answer, k=k, max_sources=max_sources,
                marginal_relevance=marginal_relevance,)
                
        bib = dict()
        if len(answer.context) < 8:
            answer_text = ("I cannot answer this question due to insufficient information.")
        else:
            qa_chain = make_chain(self.prompts.qa, cast(BaseLanguageModel, self.llm),)
            answer_text = await qa_chain.arun(context=answer.context,
                answer_length=answer.answer_length,question=answer.question,)
        
        # it happens (due to a prompt)
        if "(Example2012)" in answer_text:
            answer_text = answer_text.replace("(Example2012)", "")
            
        for c in answer.contexts:
            name = c.text.name
            citation = c.text.doc.citation
            # do check for whole key
            if name_in_text(name, answer_text):
                bib[name] = citation
        
        bib_str = "\n\n".join([f"{i+1}. ({k}): {c}" for i, (k, c) in enumerate(bib.items())])
        
        formatted_answer = f"Question: {query}\n\n{answer_text}\n"
        if len(bib) > 0:
            formatted_answer += f"\nReferences\n\n{bib_str}\n"
        
        answer.answer = answer_text
        answer.formatted_answer = formatted_answer
        answer.references = bib_str

        return answer
