#!/usr/bin/env bash
set -euo pipefail

pip install -q papermill nbconvert

# NB03 deploys a 32B LLM to the cluster Serve controller; free it on any exit.
trap 'serve shutdown -y || true' EXIT

# NB01 (doc processing) + NB02 (Ray Data ingestion → ChromaDB at /mnt/cluster_storage/vector_store).
for nb in \
  "01_(Optional)_Regular_Document_Processing_Pipeline" \
  "02_Scalable_RAG_Data_Ingestion_with_Ray_Data"; do
  papermill "notebooks/${nb}.ipynb" "/tmp/${nb}.out.ipynb" -k python3 --log-output --cwd notebooks
done

# Retrieval smoke (no LLM): fast gate on NB02's store before the GPU half.
python - <<'PY'
import chromadb
from sentence_transformers import SentenceTransformer
CHROMA_PATH = "/mnt/cluster_storage/vector_store"
COLLECTION  = "anyscale_jobs_docs_embeddings"
MODEL       = "intfloat/multilingual-e5-large-instruct"
collection = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(COLLECTION)
assert collection.count() > 0, "ChromaDB collection is empty after NB02 ingestion"
emb = SentenceTransformer(MODEL, device="cpu").encode("how to submit anyscale jobs", convert_to_numpy=True).tolist()
hits = collection.query(query_embeddings=[emb], n_results=3, include=["documents", "distances"])["ids"][0]
assert len(hits) > 0, "retrieval returned no results"
print("Retrieval smoke passed")
PY

# NB03 deploys Qwen2.5-32B (TP=4, L4) via Ray Serve. Strip its `serve shutdown` cell
# (tagged skip-in-ci) so the app PERSISTS on the cluster for NB04-06's localhost queries.
jupyter nbconvert --to notebook "notebooks/03_Deploy_LLM_with_Ray_Serve.ipynb" \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags='["skip-in-ci"]' \
  --output /tmp/03_deploy.ci.ipynb
papermill /tmp/03_deploy.ci.ipynb /tmp/03_deploy.out.ipynb -k python3 --log-output --cwd notebooks

# NB04/05/06 query the persisted localhost:8000 LLM. RAG_EVAL_N caps NB06's eval loop
# (5 requests in CI; unset = the full 63-row eval users get).
export RAG_EVAL_N=5
for nb in \
  "04_Build_Basic_RAG_Chatbot" \
  "05_Improve_RAG_with_Prompt_Engineering" \
  "06_(Optional)_Evaluate_RAG_with_Online_Inference"; do
  papermill "notebooks/${nb}.ipynb" "/tmp/${nb}.out.ipynb" -k python3 --log-output --cwd notebooks
done

# Free the 4xL4 before NB07 builds its OWN 32B batch LLM. serve shutdown is async, so
# wait (bounded poll) for the GPUs to be reclaimed to avoid contention at NB07 vLLM init.
serve shutdown -y
python - <<'PY'
import time
try:
    import ray
    ray.init(address="auto", log_to_driver=False)
    freed = False
    for _ in range(30):  # up to ~5 min
        if ray.available_resources().get("GPU", 0) >= 4:
            freed = True
            break
        time.sleep(10)
    print("4 GPUs reclaimed" if freed else "WARN: <4 GPUs free after wait; proceeding")
except Exception as e:
    print(f"WARN: GPU-reclaim poll skipped ({e}); proceeding")
PY

# NB07: self-contained Ray Data batch inference over the eval CSV (own L4/TP=4 vLLM).
papermill "notebooks/07_Evaluate_RAG_with_Ray_Data_LLM_Batch_inference.ipynb" \
  "/tmp/07_batch.out.ipynb" -k python3 --log-output --cwd notebooks
