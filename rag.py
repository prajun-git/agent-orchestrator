from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_DIR = Path("knowledge")
INDEX_DIR = "chroma_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_DISTANCE = 1.3  # chunks farther than this are treated as "not relevant"

_store = None


def _get_store() -> Chroma:
    """Load the embedding model and vector store once, then reuse them."""
    global _store
    if _store is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _store = Chroma(
            collection_name="knowledge",
            embedding_function=embeddings,
            persist_directory=INDEX_DIR,
        )
    return _store


def build_index() -> int:
    """Read every .txt/.md file in knowledge/, chunk it, embed it, store it."""
    docs = [
        Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
        for p in sorted(KNOWLEDGE_DIR.glob("*"))
        if p.suffix.lower() in {".txt", ".md"}
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = splitter.split_documents(docs)
    store = _get_store()
    store.reset_collection()  # rebuild from scratch so re-running never duplicates
    store.add_documents(chunks)
    return len(chunks)


def search_knowledge(query: str, k: int = 3) -> str:
    """Return up to k relevant chunks, dropping anything too far from the query."""
    k = max(1, min(int(k), 5))  # models sometimes send numbers as strings; also cap the size
    results = _get_store().similarity_search_with_score(query, k=k)
    relevant = [(doc, score) for doc, score in results if score <= MAX_DISTANCE]
    if not relevant:
        return (
            "No relevant documents found for this query. If the query was short or phrased "
            "as an instruction, retry once with a full question that names the topic and "
            "the organization, for example 'What is the <topic> policy of <organization>?'"
        )
    return "\n\n".join(
        f"[{doc.metadata['source']}] (distance {score:.2f})\n{doc.page_content}"
        for doc, score in relevant
    )