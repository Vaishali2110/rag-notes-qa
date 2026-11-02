# RAG v2: Hybrid Search \+ Re-ranking \+ Groq

A small Retrieval-Augmented Generation (RAG) project that answers questions from a short text document, using only the retrieved content. It improves a basic embeddings-only RAG by adding keyword search, a re-ranker, and a real LLM, and it measures what actually helped.

**Author:** Vaishali Bishtaniya

## What it does

Given a question, the system:

1. finds the most relevant chunks of the document (hybrid search),  
2. re-orders them with a cross-encoder re-ranker and drops weak ones,  
3. asks an LLM to answer **only** from those chunks, citing chunk numbers,  
4. says "I don't know based on the notes" when the document has no answer.

## Data

`data/combined_notes.txt`: a short text document (about 9 paragraph-sized chunks) covering RAG concepts (embeddings, vector databases, the RAG pipeline) and an overview of annotation for waste-management AI. It is a small demo corpus, not a large dataset.

## Pipeline

```
Question
   |
   +--> Embedding search (all-MiniLM-L6-v2 + Chroma) --+
   |                                                   +--> Reciprocal Rank Fusion --> top 5
   +--> Keyword search (BM25, stopwords removed) ------+
                                                              |
                                              Cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2)
                                                              |
                                             keep chunks with score > 0 (max 4)
                                                              |
                                      LLM via Groq (openai/gpt-oss-20b, temperature 0)
                                                              |
                                          Answer with chunk citations or "I don't know"
```

## Tech stack

- Python, Google Colab  
- LangChain text splitter (paragraph-based chunks, size 300, overlap 50\)  
- Hugging Face sentence-transformers (embeddings and cross-encoder re-ranker)  
- ChromaDB (vector store)  
- rank\_bm25 (keyword search)  
- Groq API (LLM)

## Results

I wrote a 6-question test set myself and chose the correct chunk for each question.

| Method | Correct chunk ranked first (top-1) | Correct chunk in top 2 |
| :---- | :---- | :---- |
| Embeddings only (v1) | 4 / 6 | 5 / 6 |
| Hybrid (embeddings \+ BM25) | 5 / 6 | 6 / 6 |
| Hybrid \+ re-ranker | 5 / 6 | 5 / 6 |

**Takeaways**

- Hybrid search helped on this set. It caught a question that embeddings alone missed completely.  
- The re-ranker did not improve these numbers. When I looked into the one case where it pushed out a correct chunk, three chunks (7, 8, 9\) all discussed annotation quality with almost equal scores, so my label was ambiguous.  
- With only 6 questions, one question changes the score by about 17%. These numbers are an indication, not proof.

**Manual checks of the final answers**

- A false-premise question ("List the 5 steps of the RAG pipeline") is corrected: the system answers that the notes list 3 steps.  
- A question about waste-sorting uses chunks from several parts of the document and cites them.  
- An out-of-scope question (a cricket result) gets "I don't know based on the notes".

## Problems I hit and how I solved them

1. **BM25 returned the wrong chunk.** "similarity?" and "similarity," were different tokens, and common words inflated scores. Fixed with a tokenizer that removes punctuation and stopwords.  
2. **Ambiguous test label.** Several chunks could be correct for one question. Lesson: a golden set should allow more than one acceptable chunk.  
3. **A relevant chunk was cut by top-k.** The re-ranker scored it positive, but it ranked 4th and k=3 dropped it. Fixed with k=4 (trade-off: one irrelevant chunk sometimes comes along).  
4. **Over-refusal.** The model said "I don't know" to the false-premise question. Fixed by splitting the prompt rules into "partial answer" and "nothing relevant".  
5. **`model_not_found` error.** Listed the models available on my account and kept the model name in one `MODEL` variable.  
6. **`NameError` after session restart.** Cells must run top to bottom; all libraries are installed in the first cell.  
7. **API key safety.** The key is stored in Colab Secrets, never in the code.

## Repository structure

```
rag_notes_qa/
  README.md
  data/
    combined_notes.txt
  notebooks/
    rag_notes_qa.ipynb       (v1: basic RAG, embeddings only)
    rag_notes_qa_v2.ipynb    (v2: hybrid search + re-ranking + Groq)
```

## How to run

1. Keep the folder structure above in Google Drive, in a folder named `rag_notes_qa` (the notebook reads `/content/drive/MyDrive/rag_notes_qa/data/combined_notes.txt`).  
2. Create a free Groq API key and add it in Colab **Secrets** as `GROQ_API_KEY` (turn on Notebook access).  
3. Open `notebooks/rag_notes_qa_v2.ipynb` and use **Runtime \> Restart session and run all**.  
4. If the model name is not available on your account, list your models and change the `MODEL` variable.

## Limitations

- Tiny corpus and a 6-question test set, so results are indicative only.  
- The score threshold (0) and k (4) were chosen by hand, not tuned.  
- Answer quality was checked manually, not with an automated faithfulness metric.  
- Notebook only: no API or deployment yet.

## Next steps

- Build a larger golden set that accepts multiple correct chunks.  
- Add automated evaluation (for example RAGAS or DeepEval).  
- Tune the threshold and k.  
- Wrap the pipeline in a FastAPI service and deploy it.