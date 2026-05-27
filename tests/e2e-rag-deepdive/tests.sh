#!/usr/bin/env bash
set -euo pipefail

pip install -q papermill

# NB01 (doc processing) + NB02 (Ray Data ingestion; persists ChromaDB to /mnt/cluster_storage/vector_store).
notebooks=(
  "01_(Optional)_Regular_Document_Processing_Pipeline"
  "02_Scalable_RAG_Data_Ingestion_with_Ray_Data"
)
for nb in "${notebooks[@]}"; do
  papermill "notebooks/${nb}.ipynb" "/tmp/${nb}.out.ipynb" -k python3 --log-output --cwd notebooks
done

# NB03-07 skipped: each needs a Qwen2.5-32B LLM on 4x L4 (serve/chatbot/eval) — beyond the CI floor.

# Retrieval smoke (no LLM): query NB02's persisted store with the same e5 model; assert hits.
python - <<'PY'
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "/mnt/cluster_storage/vector_store"
COLLECTION  = "anyscale_jobs_docs_embeddings"
MODEL       = "intfloat/multilingual-e5-large-instruct"
QUERY       = "how to submit anyscale jobs"
TOP_K       = 3

collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
count = collection.count()
print(f"vector store '{COLLECTION}' holds {count} vectors")
assert count > 0, "ChromaDB collection is empty after NB02 ingestion"

embedding = SentenceTransformer(MODEL, device="cpu").encode(QUERY, convert_to_numpy=True).tolist()
results = collection.query(
    query_embeddings=[embedding], n_results=TOP_K, include=["documents", "distances"]
)
hits = results["ids"][0]
print(f"top-{TOP_K} retrieval for {QUERY!r}: {len(hits)} hit(s)")
assert len(hits) > 0, "retrieval returned no results"
print("Retrieval smoke passed")
PY
