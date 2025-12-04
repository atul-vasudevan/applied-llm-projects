from typing import List
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


def load_public_docs(limit: int = 500) -> List[Document]:
    """
    Load a small open dataset (AG News) and convert
    it into LangChain Documents.
    """
    ds = load_dataset("ag_news", split="train")

    docs: List[Document] = []
    for i, row in enumerate(ds):
        if i >= limit:
            break
        text = row["text"]
        label = row["label"]
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": f"ag_news_{i}",
                    "label": int(label),
                },
            )
        )
    return docs


def build_vectorstore(limit: int = 500) -> FAISS:
    """
    Build a FAISS index over the public dataset.
    """
    docs = load_public_docs(limit=limit)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vs = FAISS.from_documents(docs, embedding=embeddings)
    return vs
