import os
import ray
import uuid
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from optimum.bettertransformer import BetterTransformer

from util.utils import generate_output_path
from util.embedding_util import ComputeEmbeddings

# Step 1: Setup model defaults
HF_MODEL_NAME = "thenlper/gte-large"
# Some Hugging Face models require a token for access; if you choose one of these models, replace the following with your token.
HF_TOKEN = "<REPLACE_WITH_YOUR_HUGGING_FACE_USER_TOKEN>"

NUM_MODEL_INSTANCES = 4
# The name of Dataset column with the input text.
TEXT_COLUMN_NAME = "text"
OUTPUT_PATH = generate_output_path(os.environ.get("ANYSCALE_ARTIFACT_STORAGE"), HF_MODEL_NAME)

if ray.is_initialized():
    ray.shutdown()
ray.init(
    runtime_env={
        "env_vars": {
            "HF_TOKEN": HF_TOKEN,

            # Suppress noisy logging from external libraries
            "TOKENIZERS_PARALLELISM": "false",
            "COMET_API_KEY": "",
        },
    }
)

# Step 2: Read data files
ds = ray.data.read_text("s3://anonymous@air-example-data/wikipedia-text-embeddings-100.txt")

# Step 3: Preprocess (Chunk) Input Text
CHUNK_SIZE = 512
words_to_tokens = 1.2
num_tokens = int(CHUNK_SIZE // words_to_tokens)
num_words = lambda x: len(x.split())

splitter = RecursiveCharacterTextSplitter(
    chunk_size=num_tokens,
    keep_separator=True,
    length_function=num_words,
    chunk_overlap=0,
)


def chunk_row(row):
    """Main chunking logic."""
    if not isinstance(row[TEXT_COLUMN_NAME], str):
        print(
            "Found row with missing input text, "
            f"skipping embeddings computation for this row: {row}"
        )
        return []

    length = num_words(row[TEXT_COLUMN_NAME]) * 1.2
    if length < 20 or length > 4000:
        return []
    chunks = splitter.split_text(row[TEXT_COLUMN_NAME])

    new_rows = [
        {
            "id": str(uuid.uuid4()),
            **row,
            TEXT_COLUMN_NAME: chunk,
        } for chunk in chunks
        # Only keep rows with valid input text
        if isinstance(chunk, str)
    ]
    return new_rows


chunked_ds = ds.flat_map(chunk_row)

embedded_ds = chunked_ds.map_batches(
    ComputeEmbeddings,
    # Total number of GPUs to use.
    concurrency=NUM_MODEL_INSTANCES,
    # Size of batches passed to embeddings actor. Set the batch size to as large possible
    # without running out of memory. If you encounter CUDA out-of-memory errors, decreasing
    # batch_size may help.
    batch_size=25,
    fn_constructor_kwargs={
        "text_column_name": TEXT_COLUMN_NAME,
        "model_name": HF_MODEL_NAME,
        "device": "cuda",
        "chunk_size": CHUNK_SIZE,
    },
    # 1 GPU for each actor.
    num_gpus=1,
    # Reduce GPU idle time.
    max_concurrency=2,
)

# Write results to cloud storage
embedded_ds.write_parquet(OUTPUT_PATH, try_create_dir=False)

print(f"Computed embeddings are written into {OUTPUT_PATH}.")
