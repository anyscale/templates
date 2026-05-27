#!/usr/bin/env bash
set -euo pipefail

pip install -q papermill

# Run the RAG ingestion notebooks via papermill (ipykernel runs their !/%%bash cells
# natively). NB01 = regular doc-processing demo; NB02 = scalable Ray Data ingestion that
# persists the ChromaDB vector store to /mnt/cluster_storage/vector_store.
# They use relative paths under notebooks/, so execute with that as the working dir.
notebooks=(
  "01_(Optional)_Regular_Document_Processing_Pipeline"
  "02_Scalable_RAG_Data_Ingestion_with_Ray_Data"
)
for nb in "${notebooks[@]}"; do
  papermill "notebooks/${nb}.ipynb" "/tmp/${nb}.out.ipynb" -k python3 --log-output --cwd notebooks
done

# NB03-07 stay out of CI — each needs the expensive 4x L4 LLM, beyond the CI floor:
#   03 Deploy LLM with Ray Serve        - boots the 4x L4 model endpoint
#   04 Build Basic RAG Chatbot          - needs the served LLM
#   05 Improve RAG with Prompt Engineering - needs the served LLM
#   06 Evaluate RAG (Online Inference)  - online inference against the served LLM
#   07 Evaluate RAG (Ray Data Batch)    - LLM batch inference

# Retrieval smoke: query the ChromaDB that NB02 persisted, using the SAME e5 model NB02
# embedded with, and assert top-k returns hits. No LLM involved.
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
