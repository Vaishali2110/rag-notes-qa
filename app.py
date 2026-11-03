import os
import re

from fastapi import FastAPI
from groq import Groq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

DATA_PATH = os.environ.get("DATA_PATH", "data/combined_notes.txt")
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# 1) Load and chunk the document
with open(DATA_PATH, "r", encoding="utf-8") as f:
    doc_text = f.read()
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50, separator="\n\n")
chunks = splitter.split_text(doc_text)

# 2) Embeddings and vector store
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_texts(chunks, embedding_model)

# 3) Keyword search (BM25) with punctuation and stopwords removed
STOP = {"what", "is", "the", "a", "an", "of", "to", "and", "in", "how", "does", "do",
        "for", "are", "it", "this", "that", "with", "as", "by", "on"}


def tok(text):
    return [w for w in re.findall(r"[a-z0-9\-]+", text.lower()) if w not in STOP]


bm25 = BM25Okapi([tok(c) for c in chunks])


# 4) Hybrid search: combine both rankings with Reciprocal Rank Fusion
def hybrid_search(query, k=2, rrf_k=60):
    sem_docs = vector_db.similarity_search(query, k=len(chunks))
    sem_rank = [chunks.index(d.page_content) for d in sem_docs]
    kw_scores = bm25.get_scores(tok(query))
    kw_rank = sorted(range(len(chunks)), key=lambda i: kw_scores[i], reverse=True)
    fused = {}
    for ranking in (sem_rank, kw_rank):
        for pos, i in enumerate(ranking):
            fused[i] = fused.get(i, 0) + 1 / (rrf_k + pos + 1)
    return sorted(fused, key=fused.get, reverse=True)[:k]


# 5) Re-ranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_rerank(query, fetch=5, k=2):
    cand = hybrid_search(query, k=fetch)
    scores = reranker.predict([(query, chunks[i]) for i in cand])
    return sorted(zip(cand, scores), key=lambda x: x[1], reverse=True)[:k]


# 6) Answer generation with Groq (key comes from an environment variable)
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("Set the GROQ_API_KEY environment variable.")
client = Groq(api_key=api_key)


def answer(query, k=4, min_score=0.0):
    hits = hybrid_rerank(query, fetch=6, k=k)
    keep = [(i, s) for i, s in hits if s > min_score]
    if not keep:
        return "I don't know based on the notes.", []
    context = "\n\n".join(f"[chunk {i + 1}] {chunks[i]}" for i, _ in keep)
    prompt = f"""Answer using only the context below.
Rules:
1. If the context fully or partly answers the question, give what the context says. If the question assumes something different from the context (for example a different number), say what the context actually states.
2. Only if the context has nothing relevant, say "I don't know based on the notes."
3. Mention the chunk numbers you used.

Context:
{context}

Question: {query}
Answer:"""
    r = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content, [i + 1 for i, _ in keep]


# 7) API
app = FastAPI(title="RAG notes QA")


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(chunks)}


@app.get("/ask")
def ask(question: str):
    text, used = answer(question)
    return {"answer": text, "chunks_used": used}
