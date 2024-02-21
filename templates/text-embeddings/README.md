# Computing Text Embeddings with Ray
The Python file `ray_embeddings.py` is used to generate text embeddings using a pre-trained HuggingFace model. We also provide `ray_embeddings_notebook.ipynb` which has the same code logic, but in notebook form, in case you want to tinker and customize the logic.
The high-level steps involved are:

0. Read in raw data from user-provided input parquet file paths.
1. Preprocess the raw input text by chunking to rows of 512 words.
2. Compute embeddings using a pre-trained HuggingFace model (default model is [`thenlper/gte-large`](https://huggingface.co/thenlper/gte-large)).
3. Write computed embeddings to parquet files in the user-provided output path.

![Embeddings Computation Overview](diagram.jpg "Embeddings Computation Overview")

## Running the code
The provided script will work as-is; no need to modify anything the underlying script, unless you want to customize the logic to meet your specific requirements.

Sample usage:
```
# Generate emeddings for a sample single-file parquet file containing 10 rows from the `falcon-refinedweb` HuggingFace dataset, using 1 GPU.
python ray_embeddings.py \
    --input-paths "sample_input/sample-input.parquet" \
    --output-path  "/mnt/cluster_storage/embeddings-output/" \
    --num-gpu-workers 1
    --column-name <text column name> # required for multi-column input files
```

Args for the script:
- `--input-paths`: comma-separated string of input file paths containing raw text data, e.g. S3 bucket ("s3://..."), local directory, etc. Accepted file types: `.json`, `.jsonl`, `.parquet`, and text files.
- `--output-paths`: output path for computed embeddings, e.g. S3 bucket ("s3://..."), local directory, etc.
- `--hf-model-name`: name of pre-trained Hugging Face model used for computing embeddings. Default is `thenlper/gte-large`
- `--num-gpu-workers`: number of GPU workers to use; this is typically the number of GPUs in the cluster
- `--column-name`: name of column containing the text to generate embeddings. Required if the input files contain multiple columns.
- `--output-text-embeddings-only`: optional; if used, the output files contain only text and embedding vector columns. Otherwise (by default), the output files will contain all columns from the input file, plus the embedding vectors in a column named `"values"`.

To read back the computed embeddings, you can also use Ray Data:
```
>>> import ray
>>> ds = ray.data.read_parquet("<embeddings_output_path>")
>>> ds.take(5)
```

## Input / Output File Formats
The script currently accepts `.json`, `.jsonl`, `.parquet`, and text files.

The **input files** should all follow the same schema:
- A single column containing the raw input text. For a single-column file, the text column will automatically be inferred. For files with multiple columns, the text column should be specified with the `--column-name` flag.
- Zero or more other columns, which will also be included in the output file unless the `--output-text-embeddings-only` is used.

The **output files** will follow the following schema:
- `id (string)`: UUID4 representing a unique ID for this row/embedding vector
- `values (list[float32])`: embedding vector
- a column containing the chunked text with the same column name as the raw text in the input file; this is the same as `--column-name` if provided (or the only column in a single-column input file)
- all other columns from input file (if `--output-text-embeddings-only` is used, these columns are omitted).

For example, if the input file contains columns `["text", "rank", "is_dupe"]`:
- Without `--output-text-embeddings-only` flag: output file would have columns `["id", "values", "text", "rank", "is_dupe"]`.
- With `--output-text-embeddings-only` flag: output file would have columns `["id", "text", "values"]`.

## Sample Input / Output Files
We have also provided **sample parquet files** under `sample_input/` and corresponding output files under `sample_output/` (both of the input files contain the same raw text so the output embeddings are the same, but the `-multi-column.parquet` file has additional metadata).

Examining the sample input file:

```
>>> sample_input = ray.data.read_parquet("sample_input/sample-input.parquet")

>>> sample_input
Dataset(num_blocks=768, num_rows=10, schema={content: string})
```

Examining the sample (expected) output file:

```
>>> sample_output = ray.data.read_parquet("sample_output/sample-output.parquet")

>>> sample_output
Dataset(
   num_blocks=768,
   num_rows=15,
   schema={
      id: string,
      content: string,
      url: string,
      dump: string,
      segment: string,
      __index_level_0__: int64,
      values: list<item: float>
   }
)
```

Examining the sample (expected) output file when using the `--output-text-embeddings-only` flag:
```
>>> sample_output_embeddings_only = ray.data.read_parquet("sample_output/sample-output-embeddings-only.parquet")

>>> sample_output_embeddings_only
Dataset(
   num_blocks=768,
   num_rows=15,
   schema={id: string, content: string, values: list<item: float>}
)
```
