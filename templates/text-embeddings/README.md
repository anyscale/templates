# Computing Text Embeddings with Ray Data

**⏱️ Time to complete**: 10 min

This template shows you how to:
1. Read in data from files on cloud storage using Ray Data.
2. Chunk the raw input text using Ray Data and LangChain.
3. Compute embeddings using a pre-trained HuggingFace model, and write the results to cloud storage.

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/text-embeddings/assets/diagram.jpg"/>

For a Python script version of the `.ipynb` notebook used for the workspace template, refer to `main.py`.

**Note:** This tutorial is run within a workspace. See [Introduction to Workspaces](https://docs.anyscale.com/examples/intro-workspaces) for more details.



## Step 1: Setup model defaults

First, install additional required dependencies using `pip`.


```python
!pip install -q langchain==0.1.17 optimum==1.19.2 && echo 'Install complete!'
```


```python
# Import utilities used during embeddings computation later (Step 4). This cell copies
# the implementation into a local file located at `util/embedding_util.py`, which
# you can choose to customize for your own use case.
import inspect
import os
import shutil
import ray.anyscale.data.embedding_util

module_file_path = inspect.getfile(ray.anyscale.data.embedding_util)
copy_dest_path = 'util/embedding_util.py'
if not os.path.isfile(copy_dest_path):
    # Only copy the file if it doesn't exist, so we don't overwrite any existing changes.
    out_path = shutil.copy(module_file_path, copy_dest_path)
    print(f"File copied to {out_path}")
```

Let's import the dependencies we will use in this template.


```python
import ray
import uuid
from langchain.text_splitter import RecursiveCharacterTextSplitter

from util.utils import generate_output_path
```

Set up default values that will be used in the embeddings computation workflow:
* The Hugging Face embedding model to use for computing embeddings. You can choose from one of the models on the [Massive Text Embedding Benchmark (MTEB) leaderboard](https://huggingface.co/spaces/mteb/leaderboard). This workspace template has been tested with the following models:
  * [`thenlper/gte-large`](https://huggingface.co/thenlper/gte-large) (the default model)
  * [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
  * [`WhereIsAI/UAE-Large-V1`](https://huggingface.co/WhereIsAI/UAE-Large-V1)
  * [`intfloat/multilingual-e5-large-instruct`](https://huggingface.co/intfloat/multilingual-e5-large-instruct)
  * [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5)
* The number of embedding model instances to run concurrently (when using GPUs, this is the total number of GPUs to use).
* The output path where results will be written as parquet files.


```python
HF_MODEL_NAME = "thenlper/gte-large"
# Optional: The models listed above do not require a Hugging Face token, 
# so there is no need to set this variable in that case.
# If your model requires a token for access, replace the following with your user token.
OPTIONAL_HF_TOKEN_FOR_MODEL = "<OPTIONAL_HUGGING_FACE_USER_TOKEN>"
NUM_MODEL_INSTANCES = 4
OUTPUT_PATH = generate_output_path(os.environ.get("ANYSCALE_ARTIFACT_STORAGE"), HF_MODEL_NAME)
```

Start up Ray, using the Hugging Face token as an environment variable so that it's made available to all nodes in the cluster.


```python
if ray.is_initialized():
    ray.shutdown()
ray.init(
    runtime_env={
        "env_vars": {
            # Set the user token on all nodes.
            "HF_TOKEN": OPTIONAL_HF_TOKEN_FOR_MODEL,

            # Suppress noisy logging from external libraries
            "TOKENIZERS_PARALLELISM": "false",
            "COMET_API_KEY": "",
        },
    }
)
```

## Step 2: Read data files

Use Ray Data to read in your input data from some sample text.


```python
sample_text = [
    "This discussion is a very interesting experimental model for the elucidation of plaque rupture in acute coronary syndromes. The knowledge exists that there is a series of steps in develoiping atheromatous plaque. We also know that platelets and endothelium are the location of this pathological development. We don’t know exactly the role or mechanism of the contribution of hyperlipidemia, and what triggers plaque rupture. This work reported is an experimental rabbit model that sheds light on the triggering of plaque rupture.",
    "the long (8-month) preparatory period. In addition, there is a need to replicate the findings of Constantinides and Chakravarti(13) from 30 years ago because of the biological variability of rabbit strains and RVV.', 'Methods', 'Of the 12 rabbits that died during the preparatory period, 5 were in group II, 2 in group III, and 5 in group IV. Seven of the 12 rabbits that died prematurely underwent an autopsy, and none had evidence of plaque disruption or arterial thrombosis. The causes of death included respiratory infection and liver failure from lipid infiltration.",
    "The triggering agents RVV (Sigma Chemical Co) and histamine (Eli Lilly) were administered according to the method of Constantinides and Chakravarti.(13) RVV (0.15 mg/kg) was given by intraperitoneal injection 48 and 24 hours before the rabbits were killed. Thirty minutes after each RVV injection, histamine (0.02 mg/kg) was administered intravenously through an ear vein. Rabbits were killed by an overdose of intravenous pentobarbital and potassium chloride. The aorta and iliofemoral arteries were dissected and excised, and the intimal surface was exposed by an anterior longitudinal incision of the vessel.",
    "The total surface area of the aorta, from the aortic arch to the distal common iliac branches, was measured. The surface area covered with atherosclerotic plaque and the surface area covered with antemortem thrombus were then determined. Images of the arterial surface were collected with a color charge-coupled device camera (TM 54, Pulnix) and digitized by an IBM PC/AT computer with a color image processing subsystem. The digitized images were calibrated by use of a graticule, and surface areas were measured by use of a customized quantitative image analysis package.",
    "Tissue samples (1 cm in length) were taken from the thoracic aorta, 3 and 6 cm distal to the aortic valve; from the abdominal aorta, 7 and 4 cm proximal to the iliac bifurcation; and from the iliofemoral arteries. and prepared for and examined by light microscopy and they were examined by quantitative colorimetric assay. Electron microscopy was also carried out with a Hitachi 600 microscope.",
]
ds = ray.data.from_items(sample_text)

# View one row of the Dataset.
ds.take(1)
```

## Step 3: Preprocess (Chunk) Input Text

Use Ray Data and LangChain to chunk the raw input text. We use LangChain's `RecursiveCharacterTextSplitter` to handle the chunking within `chunk_row()`. The chunking method and parameters can be modified below to fit your exact use case.


```python
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

def chunk_row(row, text_column_name):
    """Main chunking logic."""
    if not isinstance(row[text_column_name], str):
        print(
            "Found row with missing input text, "
            f"skipping embeddings computation for this row: {row}"
        )
        return []

    length = num_words(row[text_column_name]) * 1.2
    if length < 20 or length > 4000:
        return []
    chunks = splitter.split_text(row[text_column_name])

    new_rows = [
        {
            "id": str(uuid.uuid4()),
            **row,
            text_column_name: chunk,
        } for chunk in chunks
        # Only keep rows with valid input text
        if isinstance(chunk, str)
    ]
    return new_rows

```

We use Ray Data's `flat_map()` method to apply the chunking function we defined above to each row of the input dataset. We use `flat_map()` as opposed to `map()` because the `chunk_row()` function may return more than one output row for each input row.

*Note*: Because Ray Datasets are executed in a lazy and streaming fashion, running the cell below will not trigger execution because the dataset is not being consumed yet. See the [Ray Data docs](https://docs.ray.io/en/latest/data/data-internals.html#streaming-execution]) for more details.


```python
chunked_ds = ds.flat_map(
    chunk_row,
    # Pass keyword arguments to the chunk_row function.
    fn_kwargs={"text_column_name": "item"},
)
```

## Step 4: Compute Embeddings

We define the `ComputeEmbeddings` class to compute embeddings using a pre-trained Hugging Face model. The full implementation
is in `util/embedding_util.py`, which you can choose to customize for your use case.

Next, apply batch inference for all input data with the Ray Data [`map_batches`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html) method. Here, you can easily configure Ray Data to scale the number of model instances.


```python
from util.embedding_util import ComputeEmbeddings

embedded_ds = chunked_ds.map_batches(
    ComputeEmbeddings,
    # Total number of embedding model instances to use.
    concurrency=NUM_MODEL_INSTANCES,
    # Size of batches passed to embeddings actor. Set the batch size to as large possible
    # without running out of memory. If you encounter out-of-memory errors, decreasing
    # batch_size may help. 
    batch_size=1, 
    fn_constructor_kwargs={
        "text_column_name": "item",
        "model_name": HF_MODEL_NAME,
        "device": "cpu",
        "chunk_size": CHUNK_SIZE,
    },
)
```

Run the following cell to trigger execution of the Dataset and view the computed embeddings:


```python
embedded_ds.take_all()
```

### Scaling to a larger dataset
In the example above, we computed embeddings for a Ray Dataset with 5 sample rows. Next, let's explore how to scale to a larger dataset from files stored in cloud storage.

Run the following cell to create a Dataset from a text file stored on S3. This Dataset has 100 rows, with each row containing a single string in the `text` column.


```python
ds = ray.data.read_text("s3://anonymous@air-example-data/wikipedia-text-embeddings-100.txt")
ds.take(1)
```

Apply the same chunking process on the new Dataset.


```python
chunked_ds = ds.flat_map(
    chunk_row,
    # Pass keyword arguments to the chunk_row function.
    fn_kwargs={"text_column_name": "text"},
)
```

### Scaling with GPUs

To use GPUs for inference in the Workspace, we can specify `num_gpus` and `concurrency` in the `ds.map_batches()` call below to indicate the number of embedding models and the number of GPUs per model instance, respectively. For example, with `concurrency=4` and `num_gpus=1`, we have 4 embedding model instances, each using 1 GPU, so we need 4 GPUs total.

You can also specify which accelerator type (for example, if your model requires a specific GPU) to use with the `accelerator_type` parameter. For a full list of supported accelerator types, see [the documentation](https://docs.ray.io/en/latest/ray-core/accelerator-types.html).

If you do not need to select a particular instance type, you can omit this parameter and select "Auto-select worker nodes" under the compute configuration to have Ray select the best worker nodes from the available types:

<img src="https://raw.githubusercontent.com/anyscale/templates/main/templates/text-embeddings/assets/ray-data-gpu.png" width="400"/>


```python
embedded_ds = chunked_ds.map_batches(
    ComputeEmbeddings,
    # Total number of GPUs to use.
    concurrency=NUM_MODEL_INSTANCES,
    # Size of batches passed to embeddings actor. Set the batch size to as large possible
    # without running out of memory. If you encounter CUDA out-of-memory errors, decreasing
    # batch_size may help.
    batch_size=25, 
    fn_constructor_kwargs={
        "text_column_name": "text",
        "model_name": HF_MODEL_NAME,
        "device": "cuda",
        "chunk_size": CHUNK_SIZE,
    },
    # 1 GPU for each actor.
    num_gpus=1,
    # Reduce GPU idle time.
    max_concurrency=2,
    # Uncomment the following line and specify a specific desired accelerator type.
    # If not specified, Ray will choose the best worker nodes from the available types.
    # accelerator_type="T4", # or "L4", "A10G", "A100", etc.
)
```

### Handling GPU out-of-memory failures
If you run into a `CUDA out of memory` error, your batch size is likely too large. Decrease the batch size as described above.

If your batch size is already set to 1, then use either a smaller model or GPU devices with more memory.

### Write results to cloud storage

Finally, write the computed embeddings out to Parquet files on S3. Running the following cell will trigger execution of the Dataset, kicking off chunking and embeddings computation for all input text.


```python
embedded_ds.write_parquet(OUTPUT_PATH, try_create_dir=False)
```

We can also use Ray Data to read back the output files to ensure the results are as expected.


```python
ds_output = ray.data.read_parquet(OUTPUT_PATH)
ds_output.take(5)
```

### Submitting to Anyscale Jobs

The script in `main.py` has the same code as this notebook; you can use `anyscale job submit` to submit the app in that file to Anyscale Jobs. Refer to [Introduction to Jobs](https://docs.anyscale.com/examples/intro-jobs/) for more details.


Run the following cell to submit a job:


```python
!anyscale job submit -- python main.py
```

## Summary

This notebook:
- Read in data from files on cloud storage using Ray Data.
- Chunked the raw input text using Ray Data and LangChain.
- Computed embeddings using a pre-trained HuggingFace model, and wrote the results to cloud storage.


