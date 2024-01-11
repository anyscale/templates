# ray-embeddings
The notebooks in this repo are used to generate embeddings from the `falcon-refinedweb` [dataset](https://huggingface.co/datasets/tiiuae/falcon-refinedweb), then upsert them to Pinecone. 

### Summary of files
- `0_chunk_raw_inputs.ipynb`: Read raw data from `falcon-refinedweb` Hugging Face dataset parquet files, split rows into chunks (chunked based on number of tokens), and write chunked rows to output parquet files.
- `1_generate_embeddings.ipynb`: Read chunked rows, generates embedding vectors (in this case using `thenlper/gte-large` model), and writes them to output parquet files.
- `2_merge_embeddings.ipynb`: Reads raw embeddings files (in the 100s of MB), merges them into files of ~1 million rows with each row group containing ~100K rows, and writes them to output parquet files.
- `3_upsert_merged_embeddings.ipynb`: Reads merged embeddings files, and upserts batches of embedding vectors using the Pinecone API.

### How to reproduce results
In order to reproduce the results, one can use the notebooks in the following order (user should fill in variables such as API keys, file/bucket paths, number of workers, etc. to their specification):

0. Run `0_chunk_raw_inputs.ipynb`. 
1. Run `1_generate_embeddings.ipynb` using the chunked outputs from step 0
2. Run `2_merge_embeddings.ipynb` using the generated embeddings from step 1
3. Run `3_upsert_merged_embeddings.ipynb` using the merged embedding files from step 2
