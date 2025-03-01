# Fine-tuning a face mask detection model with Faster R-CNN

This tutorial fine-tunes a pre-trained Faster R-CNN model from PyTorch to create a face mask detection model that detects if a person is wearing a face mask correctly, not wearing a mask, or wearing it incorrectly. This example demonstrates how to:
* Use a dataset from Kaggle (with 853 annotated images in Pascal VOC format).
* Parse the Pascal VOC XML annotations with Ray Data.
* Retrieve images from S3 and attach them to our dataset.
* Set up a distributed training loop using Ray Train.
* Run inference and visualize detection results.
* Save the final trained model for later use.

Time to complete: 5 minutes

This approach leverages transfer learning for efficient object detection and scales out distributed training using Ray on Anyscale.

Here is the overview of the pipeline:

![data_processing_with_object_detection_v3.drawio.png](README_files/data_processing_with_object_detection_v3.drawio.png)

## Set up dependancies

Before proceeding, install the necessary dependencies. You have two options:

### Option 1: Build a Docker image

To set up your environment on Anyscale, you need to build a Docker image with the required dependencies. See the Anyscale docs for dependency management: https://docs.anyscale.com/configuration/dependency-management/dependency-byod/

This workspace includes the `Dockerfile`. Feel free to build the image yourself on Anyscale. 

Using the Docker image may improve the workspace spin up time and worker node load time. 


### Option 2: Install libraries directly

Alternatively, you can manually install the required libraries by following this guide:
https://docs.anyscale.com/configuration/dependency-management/dependency-development




## Set up compute resources

Set up the compute resources for the project:
* Configure the Workspace (head) node with sufficient CPU and memory for task scheduling and coordination (e.g., 8 CPUs and 16 GB of memory).
* Avoid assigning a GPU to the Workspace node, because it doesn't handle training or need GPU resources.
* Add worker nodes by specifying both CPU-based and GPU-based instances:
    - CPU nodes (e.g., 8 CPUs and 16 GB) handle general processing tasks, set autoscaling from 0-10.
    - GPU nodes (e.g., 1×T4 with 4 CPUs and 16 GB) accelerate machine learning and deep learning workloads, set autoscaling from 0-10.
* Employ this hybrid setup to optimize cost and performance by dynamically allocating tasks to the most appropriate resources.

### Benefits of using Anyscale
* Worker nodes automatically shut down when no training or inference tasks are running, eliminating idle resource costs.
* Leverage autoscaling to dynamically allocate tasks to CPU or GPU nodes based on workload demands.
* Minimize infrastructure waste by ensuring that GPU resources are only active when required for ML workloads.
* Reduce costs by leveraging `Spot instances` for training with masssive data.  Anyscale also allow fallback to on-demand instances when spot instances aren't available.

For more details on setting up compute configs, see: https://docs.anyscale.com/configuration/compute-configuration/


## Kaggle data on AWS S3 

Anyscale uploaded the Kaggle mask dataset to a publicly available AWS S3 bucket. The original dataset is from kaggle: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

The dataset is structured into three main folders: `train`, `test`, and `all`:
* `all/`:  Contains 853 samples.
* `train/` : Contains 682 samples.
* `test/`: Contains 171 samples.

Each folder contains two subfolders:

* `annotations/`: Contains the Pascal VOC XML annotation files. These files include bounding box information and class labels for each image.
* `images/`: Contains the actual image files corresponding to the annotations.

This structure helps in efficiently managing and processing the data, whether you're training or evaluating your model. The `all` folder typically aggregates all available images and annotations for ease of access.


```python

## Note: Ray train v2 will be available on public Ray very soon, but in the meantime we use this workaround
## This will be removed once train v2 is pushed
import ray
ray.shutdown()
ray.init(
    runtime_env={
        "env_vars": {
            "RAY_TRAIN_V2_ENABLED": "1",
        },
    },
)

```

    2025-02-25 20:44:45,767	INFO worker.py:1654 -- Connecting to existing Ray cluster at address: 10.0.24.49:6379...
    2025-02-25 20:44:45,778	INFO worker.py:1832 -- Connected to Ray cluster. View the dashboard at [1m[32mhttps://session-3d9pcv3n1i9c9stdktc2qjksgx.i.anyscaleuserdata.com [39m[22m
    2025-02-25 20:44:45,792	INFO packaging.py:366 -- Pushing file package 'gcs://_ray_pkg_125df4410bd8f96eff9e9eda3eb57db9dbe6eee8.zip' (2.59MiB) to Ray cluster...
    2025-02-25 20:44:45,804	INFO packaging.py:379 -- Successfully pushed file package 'gcs://_ray_pkg_125df4410bd8f96eff9e9eda3eb57db9dbe6eee8.zip'.





<div class="lm-Widget p-Widget lm-Panel p-Panel jp-Cell-outputWrapper">
    <div style="margin-left: 50px;display: flex;flex-direction: row;align-items: center">
        <div class="jp-RenderedHTMLCommon" style="display: flex; flex-direction: row;">
  <svg viewBox="0 0 567 224" fill="none" xmlns="http://www.w3.org/2000/svg" style="height: 3em;">
    <g clip-path="url(#clip0_4338_178347)">
        <path d="M341.29 165.561H355.29L330.13 129.051C345.63 123.991 354.21 112.051 354.21 94.2307C354.21 71.3707 338.72 58.1807 311.88 58.1807H271V165.561H283.27V131.661H311.8C314.25 131.661 316.71 131.501 319.01 131.351L341.25 165.561H341.29ZM283.29 119.851V70.0007H311.82C331.3 70.0007 342.34 78.2907 342.34 94.5507C342.34 111.271 331.34 119.861 311.82 119.861L283.29 119.851ZM451.4 138.411L463.4 165.561H476.74L428.74 58.1807H416L367.83 165.561H380.83L392.83 138.411H451.4ZM446.19 126.601H398L422 72.1407L446.24 126.601H446.19ZM526.11 128.741L566.91 58.1807H554.35L519.99 114.181L485.17 58.1807H472.44L514.01 129.181V165.541H526.13V128.741H526.11Z" fill="var(--jp-ui-font-color0)"/>
        <path d="M82.35 104.44C84.0187 97.8827 87.8248 92.0678 93.1671 87.9146C98.5094 83.7614 105.083 81.5067 111.85 81.5067C118.617 81.5067 125.191 83.7614 130.533 87.9146C135.875 92.0678 139.681 97.8827 141.35 104.44H163.75C164.476 101.562 165.622 98.8057 167.15 96.2605L127.45 56.5605C121.071 60.3522 113.526 61.6823 106.235 60.3005C98.9443 58.9187 92.4094 54.9203 87.8602 49.0574C83.3109 43.1946 81.0609 35.8714 81.5332 28.4656C82.0056 21.0599 85.1679 14.0819 90.4252 8.8446C95.6824 3.60726 102.672 0.471508 110.08 0.0272655C117.487 -0.416977 124.802 1.86091 130.647 6.4324C136.493 11.0039 140.467 17.5539 141.821 24.8501C143.175 32.1463 141.816 39.6859 138 46.0505L177.69 85.7505C182.31 82.9877 187.58 81.4995 192.962 81.4375C198.345 81.3755 203.648 82.742 208.33 85.3976C213.012 88.0532 216.907 91.9029 219.616 96.5544C222.326 101.206 223.753 106.492 223.753 111.875C223.753 117.258 222.326 122.545 219.616 127.197C216.907 131.848 213.012 135.698 208.33 138.353C203.648 141.009 198.345 142.375 192.962 142.313C187.58 142.251 182.31 140.763 177.69 138L138 177.7C141.808 184.071 143.155 191.614 141.79 198.91C140.424 206.205 136.44 212.75 130.585 217.313C124.731 221.875 117.412 224.141 110.004 223.683C102.596 223.226 95.6103 220.077 90.3621 214.828C85.1139 209.58 81.9647 202.595 81.5072 195.187C81.0497 187.779 83.3154 180.459 87.878 174.605C92.4405 168.751 98.9853 164.766 106.281 163.401C113.576 162.035 121.119 163.383 127.49 167.19L167.19 127.49C165.664 124.941 164.518 122.182 163.79 119.3H141.39C139.721 125.858 135.915 131.673 130.573 135.826C125.231 139.98 118.657 142.234 111.89 142.234C105.123 142.234 98.5494 139.98 93.2071 135.826C87.8648 131.673 84.0587 125.858 82.39 119.3H60C58.1878 126.495 53.8086 132.78 47.6863 136.971C41.5641 141.163 34.1211 142.972 26.7579 142.059C19.3947 141.146 12.6191 137.574 7.70605 132.014C2.79302 126.454 0.0813599 119.29 0.0813599 111.87C0.0813599 104.451 2.79302 97.2871 7.70605 91.7272C12.6191 86.1673 19.3947 82.5947 26.7579 81.6817C34.1211 80.7686 41.5641 82.5781 47.6863 86.7696C53.8086 90.9611 58.1878 97.2456 60 104.44H82.35ZM100.86 204.32C103.407 206.868 106.759 208.453 110.345 208.806C113.93 209.159 117.527 208.258 120.522 206.256C123.517 204.254 125.725 201.276 126.771 197.828C127.816 194.38 127.633 190.677 126.253 187.349C124.874 184.021 122.383 181.274 119.205 179.577C116.027 177.88 112.359 177.337 108.826 178.042C105.293 178.746 102.113 180.654 99.8291 183.44C97.5451 186.226 96.2979 189.718 96.3 193.32C96.2985 195.364 96.7006 197.388 97.4831 199.275C98.2656 201.163 99.4132 202.877 100.86 204.32ZM204.32 122.88C206.868 120.333 208.453 116.981 208.806 113.396C209.159 109.811 208.258 106.214 206.256 103.219C204.254 100.223 201.275 98.0151 197.827 96.97C194.38 95.9249 190.676 96.1077 187.348 97.4873C184.02 98.8669 181.274 101.358 179.577 104.536C177.879 107.714 177.337 111.382 178.041 114.915C178.746 118.448 180.653 121.627 183.439 123.911C186.226 126.195 189.717 127.443 193.32 127.44C195.364 127.443 197.388 127.042 199.275 126.259C201.163 125.476 202.878 124.328 204.32 122.88ZM122.88 19.4205C120.333 16.8729 116.981 15.2876 113.395 14.9347C109.81 14.5817 106.213 15.483 103.218 17.4849C100.223 19.4868 98.0146 22.4654 96.9696 25.9131C95.9245 29.3608 96.1073 33.0642 97.4869 36.3922C98.8665 39.7202 101.358 42.4668 104.535 44.1639C107.713 45.861 111.381 46.4036 114.914 45.6992C118.447 44.9949 121.627 43.0871 123.911 40.301C126.195 37.515 127.442 34.0231 127.44 30.4205C127.44 28.3772 127.038 26.3539 126.255 24.4664C125.473 22.5788 124.326 20.8642 122.88 19.4205ZM19.42 100.86C16.8725 103.408 15.2872 106.76 14.9342 110.345C14.5813 113.93 15.4826 117.527 17.4844 120.522C19.4863 123.518 22.4649 125.726 25.9127 126.771C29.3604 127.816 33.0638 127.633 36.3918 126.254C39.7198 124.874 42.4664 122.383 44.1635 119.205C45.8606 116.027 46.4032 112.359 45.6988 108.826C44.9944 105.293 43.0866 102.114 40.3006 99.8296C37.5145 97.5455 34.0227 96.2983 30.42 96.3005C26.2938 96.3018 22.337 97.9421 19.42 100.86ZM100.86 100.86C98.3125 103.408 96.7272 106.76 96.3742 110.345C96.0213 113.93 96.9226 117.527 98.9244 120.522C100.926 123.518 103.905 125.726 107.353 126.771C110.8 127.816 114.504 127.633 117.832 126.254C121.16 124.874 123.906 122.383 125.604 119.205C127.301 116.027 127.843 112.359 127.139 108.826C126.434 105.293 124.527 102.114 121.741 99.8296C118.955 97.5455 115.463 96.2983 111.86 96.3005C109.817 96.299 107.793 96.701 105.905 97.4835C104.018 98.2661 102.303 99.4136 100.86 100.86Z" fill="#00AEEF"/>
    </g>
    <defs>
        <clipPath id="clip0_4338_178347">
            <rect width="566.93" height="223.75" fill="white"/>
        </clipPath>
    </defs>
  </svg>
</div>

        <table class="jp-RenderedHTMLCommon" style="border-collapse: collapse;color: var(--jp-ui-font-color1);font-size: var(--jp-ui-font-size1);">
    <tr>
        <td style="text-align: left"><b>Python version:</b></td>
        <td style="text-align: left"><b>3.12.2</b></td>
    </tr>
    <tr>
        <td style="text-align: left"><b>Ray version:</b></td>
        <td style="text-align: left"><b>2.41.0</b></td>
    </tr>
    <tr>
    <td style="text-align: left"><b>Dashboard:</b></td>
    <td style="text-align: left"><b><a href="http://session-3d9pcv3n1i9c9stdktc2qjksgx.i.anyscaleuserdata.com" target="_blank">http://session-3d9pcv3n1i9c9stdktc2qjksgx.i.anyscaleuserdata.com</a></b></td>
</tr>

</table>

    </div>
</div>





```python
## Note: Ray train v2 will be available on public Ray very soon, but in the meantime we use this workaround
## This will be removed once train v2 is pushed

%%bash
echo "RAY_TRAIN_V2_ENABLED=1" > .env
```


```python
## Note: Ray train v2 will be available on public Ray very soon, but in the meantime we use this workaround
## This will be removed once train v2 is pushed

from dotenv import load_dotenv
load_dotenv()
```




    True



### Inspect an example image

Start by fetching and displaying an example image from the S3 storage.


```python
import io

from PIL import Image
import requests

response = requests.get("https://face-masks-data.s3.us-east-2.amazonaws.com/all/images/maksssksksss0.png")
image = Image.open(io.BytesIO(response.content))
image
```




    
![png](README_files/README_9_0.png)
    



### Inspect an annotation file (Pascal VOC Format)

PASCAL VOC is a widely recognized annotation format for object detection, storing bounding boxes, object classes, and image metadata in XML files. Its structured design and common adoption by popular detection frameworks make it a standard choice for many computer vision tasks. For more details, see: http://host.robots.ox.ac.uk/pascal/VOC/

View the annotation for the above image, which is stored in Pascal VOC XML format. 



```python
!curl "https://face-masks-data.s3.us-east-2.amazonaws.com/all/annotations/maksssksksss0.xml"
```

    
    <annotation>
        <folder>images</folder>
        <filename>maksssksksss0.png</filename>
        <size>
            <width>512</width>
            <height>366</height>
            <depth>3</depth>
        </size>
        <segmented>0</segmented>
        <object>
            <name>without_mask</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <occluded>0</occluded>
            <difficult>0</difficult>
            <bndbox>
                <xmin>79</xmin>
                <ymin>105</ymin>
                <xmax>109</xmax>
                <ymax>142</ymax>
            </bndbox>
        </object>
        <object>
            <name>with_mask</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <occluded>0</occluded>
            <difficult>0</difficult>
            <bndbox>
                <xmin>185</xmin>
                <ymin>100</ymin>
                <xmax>226</xmax>
                <ymax>144</ymax>
            </bndbox>
        </object>
        <object>
            <name>without_mask</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <occluded>0</occluded>
            <difficult>0</difficult>
            <bndbox>
                <xmin>325</xmin>
                <ymin>90</ymin>
                <xmax>360</xmax>
                <ymax>141</ymax>
            </bndbox>
        </object>
    </annotation>


Observe some key fields:


* The `<size>` contains details about the image dimensions (width, height) and color depth. For instance, the following block tells us the image is 512 pixels wide, 366 pixels tall, and has 3 color channels (e.g., RGB). 

```xml
        <size>
          <width>512</width>
          <height>366</height>
          <depth>3</depth>
        </size>
```


* Each `<object>` block describes one annotated object in the image. `<name>` is the label for that object. In this dataset, it can be "with_mask" or "without_mask". "mask_weared_incorrect":

* Each `<object>` contains a `<bndbox>` tag, which specifies the coordinates of the bounding box, the rectangle that tightly encloses the object.

  - `<xmin>` and `<ymin>` are the top-left corner of the bounding box.
  - `<xmax>` and `<ymax>` are the bottom-right corner of the bounding box.


### Parse Pascal VOC annotations



The annotation files are in XML format; however, since Ray data lacks an XML parser, read the binary files directly from S3 using `ray.data.read_binary_files`.

Then, use `parse_voc_annotation` function to extract and parse XML annotation data from a binary input stored in the `bytes` field of a dataset record. It then processes the XML structure to extract bounding box coordinates, object labels, and the filename, returning them as NumPy arrays for further use.


```python
from typing import List, Tuple
import xmltodict
import numpy as np
import ray.data
import boto3

# # Create a Ray Dataset from the S3 uri.
annotation_s3_uri = "s3://face-masks-data/train/annotations/"
ds = ray.data.read_binary_files(annotation_s3_uri)


```


```python

CLASS_TO_LABEL = {
    "background": 0,
    "with_mask": 1,
    "without_mask": 2,
    "mask_weared_incorrect": 3
}


def parse_voc_annotation(record) -> dict:
    xml_str = record["bytes"].decode("utf-8")
    if not xml_str.strip():
        raise ValueError("Empty XML string")
        
    annotation = xmltodict.parse(xml_str)["annotation"]

    # Normalize the object field to a list.
    objects = annotation["object"]
    if isinstance(objects, dict):
        objects = [objects]

    boxes: List[Tuple] = []
    for obj in objects:
        x1 = float(obj["bndbox"]["xmin"])
        y1 = float(obj["bndbox"]["ymin"])
        x2 = float(obj["bndbox"]["xmax"])
        y2 = float(obj["bndbox"]["ymax"])
        boxes.append((x1, y1, x2, y2))

    labels: List[int] = [CLASS_TO_LABEL[obj["name"]] for obj in objects]
    filename = annotation["filename"]

    return {
        "boxes": np.array(boxes),
        "labels": np.array(labels),
        "filename": filename
    }


annotations = ds.map(parse_voc_annotation)
annotations.take(2)
```

    2025-02-25 20:44:56,911	INFO dataset.py:2699 -- Tip: Use `take_batch()` instead of `take() / show()` to return records in pandas or numpy batch format.
    2025-02-25 20:44:56,917	INFO streaming_executor.py:108 -- Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-25_20-40-22_741922_2443/logs/ray-data
    2025-02-25 20:44:56,918	INFO streaming_executor.py:109 -- Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[PartitionFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(parse_voc_annotation)] -> LimitOperator[limit=2]



    Running 0: 0.00 row [00:00, ? row/s]



    - ListFiles 1: 0.00 row [00:00, ? row/s]



    - PartitionFiles 2: 0.00 row [00:00, ? row/s]



    - ReadFiles 3: 0.00 row [00:00, ? row/s]



    - Map(parse_voc_annotation) 4: 0.00 row [00:00, ? row/s]



    - limit=2 5: 0.00 row [00:00, ? row/s]


    [36m(autoscaler +19s)[0m Tip: use `ray status` to view detailed cluster status. To disable these messages, set RAY_SCHEDULER_EVENTS=0.
    [36m(autoscaler +19s)[0m [autoscaler] [4CPU-8GB] Attempting to add 1 node(s) to the cluster (increasing from 0 to 1).
    [36m(autoscaler +19s)[0m [autoscaler] Launched 1 instances.





    [{'boxes': array([[321.,  34., 354.,  69.],
             [224.,  38., 261.,  73.],
             [299.,  58., 315.,  81.],
             [143.,  74., 174., 115.],
             [ 74.,  69.,  95.,  99.],
             [191.,  67., 221.,  93.],
             [ 21.,  73.,  44.,  93.],
             [369.,  70., 398.,  99.],
             [ 83.,  56., 111.,  89.]]),
      'labels': array([1, 1, 1, 1, 1, 1, 1, 1, 2]),
      'filename': 'maksssksksss1.png'},
     {'boxes': array([[ 98., 267., 194., 383.]]),
      'labels': array([1]),
      'filename': 'maksssksksss10.png'}]



### Batch image retrieval from S3
Next, fetch images from an S3 URL based on the filenames present in the batch dictionary. For each filename, check if the file has an appropriate image extension, construct the S3 URL, and then download and convert the image to an RGB NumPy array. After that, append all the loaded images into a new key "image" within the batch dictionary. 

Note that in Ray Data, the `map_batches` method only passes the batch of data to your function, meaning you can’t directly supply additional parameters like `images_s3_url`. To work around this, use `partial` to pre-bind the `images_s3_url` argument to your `read_images` function. The `read_images` function then takes just the batch (because that’s all map_batches provides) and uses the bound URL internally to fetch images from the S3 bucket. 

Note that you can use either a `function` or a `callable class` to perform the `map` or `map_batches` transformation:
* For **functions**, Ray Data uses stateless **Ray tasks**, which are ideal for simple tasks that don’t require loading heavyweight models.
* For **classes**, Ray Data uses stateful **Ray actors**, making them well-suited for more complex tasks that involve loading heavyweight models.

For more information, see : https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html and https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html


```python
from typing import Dict
import numpy as np
from PIL import Image
from functools import partial


def read_images(images_s3_url:str, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    images: List[np.ndarray] = []
    
    for filename in batch["filename"]:
        
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            continue
            
        url = os.path.join(images_s3_url, filename)
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")  # Ensure image is in RGB.

        images.append(np.array(image))
    batch["image"] = np.array(images, dtype=object)
    return batch


# URL for training images stored in S3.
train_images_s3_url = "https://face-masks-data.s3.us-east-2.amazonaws.com/train/images/"

# Bind the URL to your image reading function.
train_read_images = partial(read_images, train_images_s3_url)

# Map the image retrieval function over your annotations dataset.
train_dataset = annotations.map_batches(train_read_images)



```

### Set up Ray Train for distributed fine-tuning / training

This section configures and runs a distributed training loop using Ray Train. The training function handles several essential steps:

* **Defining the model**: Initializes a Faster R-CNN model.
* **Configuring the optimizer and scheduler**: Sets up the optimizer and learning rate scheduler for training.
* **Running the training loop**: Iterates over epochs and batches to update model parameters.
* **Checkpointing**: Saves checkpoints, but only on the primary (rank 0) worker to avoid redundant writes.

#### Distributed training with Ray Train

When launching a distributed training job, each worker executes this training function `train_func`.

  - **Without Ray Train**: You would train on a single machine or manually configure PyTorch’s `DistributedDataParallel` to handle data splitting, gradient synchronization, and communication among workers. This setup requires significant manual coordination.

  - **With Ray Train:**. Ray Train automatically manages parallelism. It launches multiple training processes (actors), each handling its own shard of the dataset. Under the hood, Ray synchronizes gradients among workers and provides features for checkpointing, metrics reporting, and more. The parallelism primarily occurs at the batch-processing step, with each worker handling a different portion of the data.

To learn more about Ray train, see: https://docs.ray.io/en/latest/train/overview.html




```python


import os
import torch
from torchvision import models
from tempfile import TemporaryDirectory

import ray
from ray import train

from torchvision import transforms 
import tempfile
from tqdm.auto import tqdm


def train_func(config):
    # Get device
    device = ray.train.torch.get_device()

    # Define model
    model = models.detection.fasterrcnn_resnet50_fpn(num_classes=len(CLASS_TO_LABEL))
    model = ray.train.torch.prepare_model(model)
    
    # Define optimizer
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )

    # Define learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config["lr_steps"], gamma=config["lr_gamma"]
    )


    for epoch in range(config["epochs"]):
        model.train()

        # Warmup learning rate scheduler for first epoch
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=warmup_factor, total_iters=250
            )
        
        # Retrieve the training dataset shard for the current worker.
        train_dataset_shard = train.get_dataset_shard("train")
        batch_iter = train_dataset_shard.iter_batches(batch_size=config["batch_size"])
        batch_iter = tqdm(batch_iter, desc=f"Epoch {epoch+1}/{config['epochs']}", unit="batch")


        for batch_idx, batch in enumerate(batch_iter):
            inputs = [transforms.ToTensor()(image).to(device) for image in batch["image"]]
            targets = [
                {
                    "boxes": torch.as_tensor(boxes).to(device),
                    "labels": torch.as_tensor(labels).to(device),
                }
                for boxes, labels in zip(batch["boxes"], batch["labels"])
            ]
            
            # Forward pass through the model.
            loss_dict = model(inputs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
             # Backpropagation.
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # Step the learning rate scheduler.
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            # Report metrics.
            current_worker = ray.train.get_context().get_world_rank()
            metrics = {
                "losses": losses.item(),
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                **{key: value.item() for key, value in loss_dict.items()},
            }

            # Print batch metrics.
            print(f"Worker {current_worker} - Batch {batch_idx}: {metrics}")
           


        if lr_scheduler is not None:
            lr_scheduler.step()

        # Save a checkpoint on the primary worker for each epoch.
        if ray.train.get_context().get_world_rank() == 0:
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                torch.save(
                    model.module.state_dict(), os.path.join(temp_checkpoint_dir, "model.pt")
                )
                checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                train.report(metrics, checkpoint=checkpoint)
        else: # Save metrics from all workers for each epoch.
            train.report(metrics)


```

#### How train.get_dataset_shard("train") works

A shard is a partition of the overall dataset allocated to a specific worker. For example, if you have 4 workers and 10,000 images, each worker receives 2,500 images (i.e., one shard of 2,500 each).

Ray Train automatically splits your dataset into shards across multiple workers. Calling `train.get_dataset_shard("train")` returns the subset (shard) of the dataset for the current worker. Each worker trains on a different shard in parallel. This approach contrasts with a typical single-machine PyTorch setup, where you might rely on PyTorch’s DataLoader or a DistributedSampler for data distribution. For more details: https://docs.ray.io/en/latest/train/api/doc/ray.train.get_dataset_shard.html


#### Batch size

The batch size specifies how many samples each worker processes in a single forward/backward pass. For instance, a batch size of 4 means each training step processes 4 samples within that worker’s shard before performing a gradient update.  In practice, you should carefully select the batch size based on the model size and GPU memory size. 

#### Checkpointing on the primary (rank 0) worker

In this example, all workers maintain the same model parameters. They are kept in sync during updates. Therefore, by the end of each epoch (or at checkpoint time), every worker’s model state is identical. Saving checkpoints from only the primary worker (rank 0) prevents redundant or conflicting writes and ensures one clear, consistent checkpoint.

To learn more about saving and loading checkpoints, see:https://docs.ray.io/en/latest/train/user-guides/checkpoints.html

#### Reporting metrics for all worker nodes

Use `train.report` to track metrics from **all worker nodes**. Ray Train’s internal bookkeeping records these metrics, enabling you to monitor progress and analyze results after training completes. 

**Note: You will receive errors if you only report the metrics from the primary worker, a common mistake to avoid.** 

### Launch the fine-tuning / training process with TorchTrainer

Configure and initiate training using TorchTrainer from Ray Train. Be patient, as this process may take some time.

**For demonstration purposes, set `epochs` to 2, but the performace of the fine-tuned model won't be optimal.** In practice, you would typically train for 20-30 epochs to achieve a well fine-tuned model.

The `num_workers` parameter specifies how many parallel worker processes will be started for data-parallel training. Set `num_workers=2` for demonstration purposes, but in real scenarios, the setting depends on:

* Your max number of available GPUs: Each worker can be assigned to one GPU (if use_gpu=True). Hence, if you have 4 GPUs, you could set num_workers=4.
* Desired training speed: More workers can lead to faster training because Ray Train splits the workload among multiple devices or processes. If your training data is large and you have the computational resources, you can increase `num_workers` to accelerate training.




```python

from ray.train.torch import TorchTrainer


storage_path = "/mnt/cluster_storage/face-mask-experiments_v1/"
run_config = ray.train.RunConfig(storage_path=storage_path, name="face-mask-experiments_v1")

trainer = TorchTrainer(
    train_func,
    train_loop_config={
        "batch_size": 4, # ajust it based on your GPU memory, a batch size that is too large could cause OOM issue
        "lr": 0.02,
        "epochs": 2,  # You'd normally train for 20-30 epochs to get a good performance.
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "lr_steps": [16, 22],
        "lr_gamma": 0.1,
    },
    scaling_config = ray.train.ScalingConfig(num_workers=2, use_gpu=True),
    run_config = run_config,
    datasets={"train": train_dataset},
)

results = trainer.fit()

```

    [36m(TrainController pid=5760)[0m Attempting to start training worker group of size 2 with the following resources: [{'GPU': 1}] * 2


    [36m(autoscaler +2m59s)[0m [autoscaler] [1xT4:4CPU-16GB] Attempting to add 2 node(s) to the cluster (increasing from 0 to 2).
    [36m(autoscaler +2m59s)[0m [autoscaler] Launched 2 instances.


    [36m(TrainController pid=5760)[0m Retrying training worker group startup. The previous attempt encountered the following failure:
    [36m(TrainController pid=5760)[0m Traceback (most recent call last):
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/train/v2/_internal/execution/worker_group/worker_group.py", line 261, in start
    [36m(TrainController pid=5760)[0m     ray.get(pg.ready(), timeout=self._worker_group_start_timeout_s)
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    [36m(TrainController pid=5760)[0m     return fn(*args, **kwargs)
    [36m(TrainController pid=5760)[0m            ^^^^^^^^^^^^^^^^^^^
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    [36m(TrainController pid=5760)[0m     return func(*args, **kwargs)
    [36m(TrainController pid=5760)[0m            ^^^^^^^^^^^^^^^^^^^^^
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/worker.py", line 2772, in get
    [36m(TrainController pid=5760)[0m     values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
    [36m(TrainController pid=5760)[0m                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/worker.py", line 893, in get_objects
    [36m(TrainController pid=5760)[0m     ] = self.core_worker.get_objects(
    [36m(TrainController pid=5760)[0m         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [36m(TrainController pid=5760)[0m   File "python/ray/_raylet.pyx", line 3190, in ray._raylet.CoreWorker.get_objects
    [36m(TrainController pid=5760)[0m   File "python/ray/includes/common.pxi", line 85, in ray._raylet.check_status
    [36m(TrainController pid=5760)[0m ray.exceptions.GetTimeoutError: Get timed out: some object(s) not ready.
    [36m(TrainController pid=5760)[0m 
    [36m(TrainController pid=5760)[0m The above exception was the direct cause of the following exception:
    [36m(TrainController pid=5760)[0m 
    [36m(TrainController pid=5760)[0m Traceback (most recent call last):
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/train/v2/_internal/execution/controller.py", line 231, in _restart_worker_group
    [36m(TrainController pid=5760)[0m     self._worker_group.start(
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/train/v2/_internal/execution/worker_group/worker_group.py", line 264, in start
    [36m(TrainController pid=5760)[0m     raise WorkerGroupStartupTimeoutError(
    [36m(TrainController pid=5760)[0m ray.train.v2._internal.exceptions.WorkerGroupStartupTimeoutError: The worker group startup timed out after 30.0 seconds waiting for 2 workers. Potential causes include: (1) temporary insufficient cluster resources while waiting for autoscaling (ignore this warning in this case), (2) infeasible resource request where the provided `ScalingConfig` cannot be satisfied), and (3) transient network issues. Set the RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S environment variable to increase the timeout.
    [36m(TrainController pid=5760)[0m Attempting to start training worker group of size 2 with the following resources: [{'GPU': 1}] * 2


    [36m(autoscaler +3m49s)[0m [autoscaler] Cluster upscaled to {8 CPU, 1 GPU}.


    [36m(TrainController pid=5760)[0m Retrying training worker group startup. The previous attempt encountered the following failure:
    [36m(TrainController pid=5760)[0m Traceback (most recent call last):
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/train/v2/_internal/execution/worker_group/worker_group.py", line 261, in start
    [36m(TrainController pid=5760)[0m     ray.get(pg.ready(), timeout=self._worker_group_start_timeout_s)
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    [36m(TrainController pid=5760)[0m     return fn(*args, **kwargs)
    [36m(TrainController pid=5760)[0m            ^^^^^^^^^^^^^^^^^^^
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    [36m(TrainController pid=5760)[0m     return func(*args, **kwargs)
    [36m(TrainController pid=5760)[0m            ^^^^^^^^^^^^^^^^^^^^^
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/worker.py", line 2772, in get
    [36m(TrainController pid=5760)[0m     values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
    [36m(TrainController pid=5760)[0m                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/worker.py", line 893, in get_objects
    [36m(TrainController pid=5760)[0m     ] = self.core_worker.get_objects(
    [36m(TrainController pid=5760)[0m         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    [36m(TrainController pid=5760)[0m   File "python/ray/_raylet.pyx", line 3190, in ray._raylet.CoreWorker.get_objects
    [36m(TrainController pid=5760)[0m   File "python/ray/includes/common.pxi", line 85, in ray._raylet.check_status
    [36m(TrainController pid=5760)[0m ray.exceptions.GetTimeoutError: Get timed out: some object(s) not ready.
    [36m(TrainController pid=5760)[0m 
    [36m(TrainController pid=5760)[0m The above exception was the direct cause of the following exception:
    [36m(TrainController pid=5760)[0m 
    [36m(TrainController pid=5760)[0m Traceback (most recent call last):
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/train/v2/_internal/execution/controller.py", line 231, in _restart_worker_group
    [36m(TrainController pid=5760)[0m     self._worker_group.start(
    [36m(TrainController pid=5760)[0m   File "/home/ray/anaconda3/lib/python3.12/site-packages/ray/train/v2/_internal/execution/worker_group/worker_group.py", line 264, in start
    [36m(TrainController pid=5760)[0m     raise WorkerGroupStartupTimeoutError(
    [36m(TrainController pid=5760)[0m ray.train.v2._internal.exceptions.WorkerGroupStartupTimeoutError: The worker group startup timed out after 30.0 seconds waiting for 2 workers. Potential causes include: (1) temporary insufficient cluster resources while waiting for autoscaling (ignore this warning in this case), (2) infeasible resource request where the provided `ScalingConfig` cannot be satisfied), and (3) transient network issues. Set the RAY_TRAIN_WORKER_GROUP_START_TIMEOUT_S environment variable to increase the timeout.
    [36m(TrainController pid=5760)[0m Attempting to start training worker group of size 2 with the following resources: [{'GPU': 1}] * 2



    (pid=6258) Running 0: 0.00 row [00:00, ? row/s]


    [36m(SplitCoordinator pid=6258)[0m Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-25_20-40-22_741922_2443/logs/ray-data
    [36m(SplitCoordinator pid=6258)[0m Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[PartitionFiles] -> TaskPoolMapOperator[ReadFiles]


    [36m(autoscaler +4m4s)[0m [autoscaler] Cluster upscaled to {12 CPU, 2 GPU}.


    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Setting up process group for: env:// [rank=0, world_size=2]
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /home/ray/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
      0%|          | 0.00/97.8M [00:00<?, ?B/s]21)[0m 
      7%|▋         | 7.00M/97.8M [00:00<00:01, 73.1MB/s]
    100%|██████████| 97.8M/97.8M [00:00<00:00, 210MB/s]
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Moving model to device: cuda:0
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Wrapping provided model in DistributedDataParallel.
    Epoch 1/2: 0batch [00:00, ?batch/s]0.0.49.121)[0m 



    (pid=6258) Running 0: 0.00 row [00:00, ? row/s]



    (pid=6258) - ListFiles 1: 0.00 row [00:00, ? row/s]



    (pid=6258) - PartitionFiles 2: 0.00 row [00:00, ? row/s]



    (pid=6258) - ReadFiles 3: 0.00 row [00:00, ? row/s]



    (pid=6258) - Map(parse_voc_annotation)->MapBatches(partial) 4: 0.00 row [00:00, ? row/s]



    (pid=6258) - split(2, equal=True) 5: 0.00 row [00:00, ? row/s]


    [36m(SplitCoordinator pid=6258)[0m Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-25_20-40-22_741922_2443/logs/ray-data
    [36m(SplitCoordinator pid=6258)[0m Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[PartitionFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(parse_voc_annotation)->MapBatches(partial)] -> OutputSplitter[split(2, equal=True)]
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /home/ray/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
      0%|          | 0.00/97.8M [00:00<?, ?B/s]61)[0m 
    100%|██████████| 97.8M/97.8M [00:00<00:00, 175MB/s][32m [repeated 8x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Moving model to device: cuda:0
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Wrapping provided model in DistributedDataParallel.
    Epoch 1/2: 0batch [00:00, ?batch/s]0.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 0: {'losses': 2.159209644139456, 'epoch': 0, 'lr': 9.992000000000002e-05, 'loss_classifier': 1.4155222177505493, 'loss_box_reg': 0.029666345566511154, 'loss_objectness': 0.7021313309669495, 'loss_rpn_box_reg': 0.011889850438284059}
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    Epoch 1/2: 1batch [14:19, 859.43s/batch]1.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 4: {'losses': 2.068951364804239, 'epoch': 0, 'lr': 0.0004196, 'loss_classifier': 1.3302171230316162, 'loss_box_reg': 0.028953097760677338, 'loss_objectness': 0.7017674446105957, 'loss_rpn_box_reg': 0.008013721753091656}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 5batch [14:24, 75.68s/batch] [32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 8: {'losses': 1.743758442754485, 'epoch': 0, 'lr': 0.0007392799999999999, 'loss_classifier': 0.9290295839309692, 'loss_box_reg': 0.08173397183418274, 'loss_objectness': 0.698894739151001, 'loss_rpn_box_reg': 0.034100177640654525}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 9batch [14:30, 16.97s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 12: {'losses': 1.0664194183651092, 'epoch': 0, 'lr': 0.00105896, 'loss_classifier': 0.33698564767837524, 'loss_box_reg': 0.037295885384082794, 'loss_objectness': 0.6834101676940918, 'loss_rpn_box_reg': 0.00872772505913999}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 13batch [14:36,  5.02s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 16: {'losses': 1.246363154812159, 'epoch': 0, 'lr': 0.0013786400000000002, 'loss_classifier': 0.4761531352996826, 'loss_box_reg': 0.09730593860149384, 'loss_objectness': 0.6549971103668213, 'loss_rpn_box_reg': 0.01790689603835527}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 17batch [14:42,  2.29s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 20: {'losses': 1.3891869918843225, 'epoch': 0, 'lr': 0.0016983199999999997, 'loss_classifier': 0.5685698390007019, 'loss_box_reg': 0.10587328672409058, 'loss_objectness': 0.617646336555481, 'loss_rpn_box_reg': 0.097097529604049}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 21batch [14:47,  1.57s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 24: {'losses': 0.5300030227907484, 'epoch': 0, 'lr': 0.002018, 'loss_classifier': 0.09626910090446472, 'loss_box_reg': 0.05076079070568085, 'loss_objectness': 0.37309157848358154, 'loss_rpn_box_reg': 0.009881567598182539}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 25batch [14:53,  1.48s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 28: {'losses': 0.47988298222674675, 'epoch': 0, 'lr': 0.002337679999999999, 'loss_classifier': 0.18640193343162537, 'loss_box_reg': 0.10459154844284058, 'loss_objectness': 0.16510945558547974, 'loss_rpn_box_reg': 0.02378004476680107}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 29batch [14:59,  1.47s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 32: {'losses': 0.6866204038540763, 'epoch': 0, 'lr': 0.002657359999999999, 'loss_classifier': 0.2348477989435196, 'loss_box_reg': 0.18193519115447998, 'loss_objectness': 0.22592699527740479, 'loss_rpn_box_reg': 0.043910433379833164}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 33batch [15:05,  1.46s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 36: {'losses': 0.4395870606977457, 'epoch': 0, 'lr': 0.0029770399999999985, 'loss_classifier': 0.2582661211490631, 'loss_box_reg': 0.1309860348701477, 'loss_objectness': 0.04159500077366829, 'loss_rpn_box_reg': 0.0087398927289957}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 37batch [15:11,  1.44s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 40: {'losses': 0.7462944566334172, 'epoch': 0, 'lr': 0.003296719999999998, 'loss_classifier': 0.3042616546154022, 'loss_box_reg': 0.2262573540210724, 'loss_objectness': 0.18080264329910278, 'loss_rpn_box_reg': 0.03497280469783985}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 41batch [15:17,  1.48s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 44: {'losses': 0.8252588615530907, 'epoch': 0, 'lr': 0.0036163999999999988, 'loss_classifier': 0.18608947098255157, 'loss_box_reg': 0.13127654790878296, 'loss_objectness': 0.37611421942710876, 'loss_rpn_box_reg': 0.13177860833348626}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 44batch [15:21,  1.38s/batch][32m [repeated 7x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 48: {'losses': 0.7530738512144833, 'epoch': 0, 'lr': 0.003936079999999999, 'loss_classifier': 0.38285693526268005, 'loss_box_reg': 0.24190860986709595, 'loss_objectness': 0.1053139790892601, 'loss_rpn_box_reg': 0.02299437914951137}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 49batch [15:28,  1.43s/batch][32m [repeated 9x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 52: {'losses': 0.637562784429218, 'epoch': 0, 'lr': 0.00425576, 'loss_classifier': 0.2582736313343048, 'loss_box_reg': 0.28213804960250854, 'loss_objectness': 0.07799634337425232, 'loss_rpn_box_reg': 0.019154700513507518}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 53batch [15:33,  1.39s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 56: {'losses': 1.0319622760953604, 'epoch': 0, 'lr': 0.004575439999999999, 'loss_classifier': 0.38727468252182007, 'loss_box_reg': 0.4629121720790863, 'loss_objectness': 0.1348753273487091, 'loss_rpn_box_reg': 0.046900153750389605}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 57batch [15:39,  1.32s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 60: {'losses': 0.491548998593351, 'epoch': 0, 'lr': 0.0048951199999999985, 'loss_classifier': 0.21487340331077576, 'loss_box_reg': 0.23454931378364563, 'loss_objectness': 0.03517657518386841, 'loss_rpn_box_reg': 0.0069497063150612114}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 61batch [15:45,  1.40s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 64: {'losses': 0.5755679013168292, 'epoch': 0, 'lr': 0.005214799999999999, 'loss_classifier': 0.2260505110025406, 'loss_box_reg': 0.3005545139312744, 'loss_objectness': 0.0380663201212883, 'loss_rpn_box_reg': 0.010896563712306535}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 65batch [15:50,  1.47s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 68: {'losses': 1.4876211453770842, 'epoch': 0, 'lr': 0.005534479999999998, 'loss_classifier': 0.5139445066452026, 'loss_box_reg': 0.6770563721656799, 'loss_objectness': 0.19629186391830444, 'loss_rpn_box_reg': 0.10032828343860768}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 69batch [15:56,  1.40s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 72: {'losses': 0.7161760773373322, 'epoch': 0, 'lr': 0.005854159999999996, 'loss_classifier': 0.26516395807266235, 'loss_box_reg': 0.35717886686325073, 'loss_objectness': 0.06690341234207153, 'loss_rpn_box_reg': 0.02692984005934767}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 73batch [16:02,  1.41s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 76: {'losses': 0.29707123375791267, 'epoch': 0, 'lr': 0.006173839999999996, 'loss_classifier': 0.15116673707962036, 'loss_box_reg': 0.12264221906661987, 'loss_objectness': 0.019693246111273766, 'loss_rpn_box_reg': 0.0035690221871729298}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 77batch [16:07,  1.42s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 80: {'losses': 0.5542587101319927, 'epoch': 0, 'lr': 0.006493519999999997, 'loss_classifier': 0.22254659235477448, 'loss_box_reg': 0.257590651512146, 'loss_objectness': 0.052237216383218765, 'loss_rpn_box_reg': 0.02188426850830491}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 81batch [16:13,  1.48s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    Epoch 1/2: 85batch [16:19,  1.48s/batch]/home/ray/anaconda3/lib/python3.12/site-packages/torchvision/transforms/functional.py:154: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m   img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 84: {'losses': 0.6542221396319422, 'epoch': 0, 'lr': 0.006813199999999997, 'loss_classifier': 0.22948068380355835, 'loss_box_reg': 0.27416008710861206, 'loss_objectness': 0.09858870506286621, 'loss_rpn_box_reg': 0.051992663656905594}[32m [repeated 8x across cluster][0m
    Epoch 1/2: 84batch [16:18,  1.49s/batch][32m [repeated 7x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    Epoch 1/2: 86batch [16:20, 11.40s/batch]1.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    Epoch 2/2: 0batch [00:00, ?batch/s]0.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/face-mask-experiments_v1/face-mask-experiments_v1/checkpoint_2025-02-25_21-05-14.136014)



    (pid=6258) Running 0: 0.00 row [00:00, ? row/s]



    (pid=6258) - ListFiles 1: 0.00 row [00:00, ? row/s]



    (pid=6258) - PartitionFiles 2: 0.00 row [00:00, ? row/s]



    (pid=6258) - ReadFiles 3: 0.00 row [00:00, ? row/s]



    (pid=6258) - Map(parse_voc_annotation)->MapBatches(partial) 4: 0.00 row [00:00, ? row/s]



    (pid=6258) - split(2, equal=True) 5: 0.00 row [00:00, ? row/s]


    [36m(SplitCoordinator pid=6258)[0m Starting execution of Dataset. Full logs are in /tmp/ray/session_2025-02-25_20-40-22_741922_2443/logs/ray-data
    [36m(SplitCoordinator pid=6258)[0m Execution plan of Dataset: InputDataBuffer[Input] -> TaskPoolMapOperator[ListFiles] -> TaskPoolMapOperator[PartitionFiles] -> TaskPoolMapOperator[ReadFiles] -> TaskPoolMapOperator[Map(parse_voc_annotation)->MapBatches(partial)] -> OutputSplitter[split(2, equal=True)]
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    Epoch 1/2: 85batch [16:19,  1.48s/batch]/home/ray/anaconda3/lib/python3.12/site-packages/torchvision/transforms/functional.py:154: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /pytorch/torch/csrc/utils/tensor_numpy.cpp:203.)
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m   img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 0: {'losses': 0.12877509557037287, 'epoch': 1, 'lr': 0.007052959999999994, 'loss_classifier': 0.0513504222035408, 'loss_box_reg': 0.06076407432556152, 'loss_objectness': 0.012926623225212097, 'loss_rpn_box_reg': 0.003733968365477849}[32m [repeated 4x across cluster][0m
    Epoch 2/2: 1batch [09:43, 583.98s/batch]9.121)[0m 
    Epoch 1/2: 86batch [16:20, 11.40s/batch]9.121)[0m 
    Epoch 2/2: 0batch [00:00, ?batch/s]0.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    Epoch 2/2: 1batch [09:43, 583.18s/batch]1.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 4: {'losses': 0.5197629828210473, 'epoch': 1, 'lr': 0.007372639999999994, 'loss_classifier': 0.20007705688476562, 'loss_box_reg': 0.2367350459098816, 'loss_objectness': 0.05626056715846062, 'loss_rpn_box_reg': 0.02669032404381034}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 5batch [09:48, 51.75s/batch][32m [repeated 7x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 8: {'losses': 0.9957833540326164, 'epoch': 1, 'lr': 0.007692319999999994, 'loss_classifier': 0.29663464426994324, 'loss_box_reg': 0.5897737145423889, 'loss_objectness': 0.05936676636338234, 'loss_rpn_box_reg': 0.05000826238451467}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 9batch [09:55, 12.00s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 12: {'losses': 0.1288612900160225, 'epoch': 1, 'lr': 0.008011999999999995, 'loss_classifier': 0.04162019491195679, 'loss_box_reg': 0.06715895235538483, 'loss_objectness': 0.006440497003495693, 'loss_rpn_box_reg': 0.013641644813862628}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 13batch [10:00,  3.87s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 16: {'losses': 0.5442380476653401, 'epoch': 1, 'lr': 0.008331679999999996, 'loss_classifier': 0.209833025932312, 'loss_box_reg': 0.24682660400867462, 'loss_objectness': 0.0723942369222641, 'loss_rpn_box_reg': 0.015184180802089369}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 17batch [10:06,  2.01s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 20: {'losses': 1.3027624165975384, 'epoch': 1, 'lr': 0.008651359999999999, 'loss_classifier': 0.48061543703079224, 'loss_box_reg': 0.6292918920516968, 'loss_objectness': 0.11712077260017395, 'loss_rpn_box_reg': 0.07573428511255315}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 21batch [10:12,  1.50s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 24: {'losses': 0.39978632377222245, 'epoch': 1, 'lr': 0.008971039999999998, 'loss_classifier': 0.15295112133026123, 'loss_box_reg': 0.21422065794467926, 'loss_objectness': 0.018916882574558258, 'loss_rpn_box_reg': 0.013697684274465494}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 25batch [10:18,  1.45s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 28: {'losses': 0.3420356814065518, 'epoch': 1, 'lr': 0.009290719999999999, 'loss_classifier': 0.1329527497291565, 'loss_box_reg': 0.1467113047838211, 'loss_objectness': 0.04300586134195328, 'loss_rpn_box_reg': 0.0193657879033627}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 29batch [10:23,  1.47s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 32: {'losses': 0.7671442364397411, 'epoch': 1, 'lr': 0.0096104, 'loss_classifier': 0.3503096401691437, 'loss_box_reg': 0.36607494950294495, 'loss_objectness': 0.030622044578194618, 'loss_rpn_box_reg': 0.020137581700361146}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 33batch [10:29,  1.46s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 36: {'losses': 0.2347540682035889, 'epoch': 1, 'lr': 0.009930079999999999, 'loss_classifier': 0.07318788766860962, 'loss_box_reg': 0.12580545246601105, 'loss_objectness': 0.026037756353616714, 'loss_rpn_box_reg': 0.009722975440641834}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 37batch [10:35,  1.43s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 40: {'losses': 0.6598218185750764, 'epoch': 1, 'lr': 0.010249759999999998, 'loss_classifier': 0.22529754042625427, 'loss_box_reg': 0.37387943267822266, 'loss_objectness': 0.03520507365465164, 'loss_rpn_box_reg': 0.025439749464205984}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 41batch [10:41,  1.47s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 44: {'losses': 0.12546871367060505, 'epoch': 1, 'lr': 0.01056944, 'loss_classifier': 0.039722077548503876, 'loss_box_reg': 0.07755278795957565, 'loss_objectness': 0.005114097148180008, 'loss_rpn_box_reg': 0.0030797547396358175}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 45batch [10:46,  1.40s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 48: {'losses': 0.606140708131529, 'epoch': 1, 'lr': 0.01088912, 'loss_classifier': 0.2786847949028015, 'loss_box_reg': 0.27388954162597656, 'loss_objectness': 0.031748801469802856, 'loss_rpn_box_reg': 0.021817540330625704}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 49batch [10:51,  1.42s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 52: {'losses': 0.37796517849039996, 'epoch': 1, 'lr': 0.011208799999999996, 'loss_classifier': 0.11183250695466995, 'loss_box_reg': 0.23459969460964203, 'loss_objectness': 0.019085580483078957, 'loss_rpn_box_reg': 0.012447383404492965}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 53batch [10:57,  1.39s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 56: {'losses': 0.657525607529463, 'epoch': 1, 'lr': 0.011528479999999997, 'loss_classifier': 0.2245432734489441, 'loss_box_reg': 0.34751981496810913, 'loss_objectness': 0.03607484698295593, 'loss_rpn_box_reg': 0.04938764232713148}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 57batch [11:03,  1.32s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 60: {'losses': 0.5538636405393432, 'epoch': 1, 'lr': 0.01184816, 'loss_classifier': 0.1759328693151474, 'loss_box_reg': 0.3198520541191101, 'loss_objectness': 0.022622879594564438, 'loss_rpn_box_reg': 0.03545581888406983}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 61batch [11:09,  1.40s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 64: {'losses': 0.39684727394811586, 'epoch': 1, 'lr': 0.01216784, 'loss_classifier': 0.12197396159172058, 'loss_box_reg': 0.23513296246528625, 'loss_objectness': 0.011202637106180191, 'loss_rpn_box_reg': 0.028537716510219115}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 65batch [11:14,  1.46s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 68: {'losses': 0.2614563464959821, 'epoch': 1, 'lr': 0.012487519999999998, 'loss_classifier': 0.09348403662443161, 'loss_box_reg': 0.14913111925125122, 'loss_objectness': 0.009750142693519592, 'loss_rpn_box_reg': 0.009091055377360262}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 69batch [11:20,  1.39s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 72: {'losses': 0.2503427677365841, 'epoch': 1, 'lr': 0.012807199999999996, 'loss_classifier': 0.09968694299459457, 'loss_box_reg': 0.14166688919067383, 'loss_objectness': 0.004086447414010763, 'loss_rpn_box_reg': 0.004902478358417912}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 73batch [11:26,  1.41s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Worker 0 - Batch 76: {'losses': 0.23268763714891083, 'epoch': 1, 'lr': 0.013126879999999995, 'loss_classifier': 0.07860548049211502, 'loss_box_reg': 0.1416209638118744, 'loss_objectness': 0.006279502529650927, 'loss_rpn_box_reg': 0.006181701956802684}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 77batch [11:31,  1.43s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 80: {'losses': 0.3773154029795067, 'epoch': 1, 'lr': 0.013446559999999996, 'loss_classifier': 0.12946566939353943, 'loss_box_reg': 0.21541208028793335, 'loss_objectness': 0.013717522844672203, 'loss_rpn_box_reg': 0.018720121140135982}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 81batch [11:38,  1.48s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m Worker 1 - Batch 84: {'losses': 0.6083476055767996, 'epoch': 1, 'lr': 0.013766239999999996, 'loss_classifier': 0.22144314646720886, 'loss_box_reg': 0.297282338142395, 'loss_objectness': 0.052109066396951675, 'loss_rpn_box_reg': 0.03751302849321196}[32m [repeated 8x across cluster][0m
    Epoch 2/2: 85batch [11:44,  1.48s/batch][32m [repeated 8x across cluster][0m
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    [36m(RayTrainWorker pid=2263, ip=10.0.49.121)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m 
    Epoch 2/2: 86batch [11:43,  8.18s/batch]1.161)[0m 
    [36m(RayTrainWorker pid=2283, ip=10.0.21.161)[0m Checkpoint successfully created at: Checkpoint(filesystem=local, path=/mnt/cluster_storage/face-mask-experiments_v1/face-mask-experiments_v1/checkpoint_2025-02-25_21-16-58.709155)


### Inspect results when training completes


```python
import torch
import os


print("Metrics reported during training:")
print(results.metrics)

print("\nLatest checkpoint reported during training:")
print(results.checkpoint)

print("\nPath where logs are stored:")
print(results.path)

print("\nException raised, if training failed:")
print(results.error)


```

    Metrics reported during training:
    {'losses': 0.07777648572283669, 'epoch': 1, 'lr': 0.013846159999999996, 'loss_classifier': 0.024532580748200417, 'loss_box_reg': 0.04629848152399063, 'loss_objectness': 0.004773554392158985, 'loss_rpn_box_reg': 0.0021718681271640753}
    
    Latest checkpoint reported during training:
    Checkpoint(filesystem=local, path=/mnt/cluster_storage/face-mask-experiments_v1/face-mask-experiments_v1/checkpoint_2025-02-25_21-16-58.709155)
    
    Path where logs are stored:
    /mnt/cluster_storage/face-mask-experiments_v1/face-mask-experiments_v1
    
    Exception raised, if training failed:
    None


### Run inference and visualize predictions on a test image
After training, run the model on a single test image for a sanity check:

* Download an image from a URL.
* Run the model for predictions.
* Visualize the detections (bounding boxes and labels).




```python
import io
import requests
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# CLASS_TO_LABEL dictionary
CLASS_TO_LABEL = {
    "background": 0,
    "with_mask": 1,
    "without_mask": 2,
    "mask_weared_incorrect": 3
}

# Create reverse label mapping
LABEL_TO_CLASS = {v: k for k, v in CLASS_TO_LABEL.items()}

# Define colors for each category
LABEL_COLORS = {
    "with_mask": "green",
    "without_mask": "red",
    "mask_weared_incorrect": "yellow"
}

def load_image_from_url(url):
    """
    Downloads the image from the given URL and returns it as a NumPy array.
    """
    response = requests.get(url)
    response.raise_for_status()  # Raise an error if the download failed.
    image = Image.open(io.BytesIO(response.content)).convert('RGB')
    return np.array(image)

def predict_and_visualize(image_np, model, confidence_threshold=0.5):
    """Run model prediction on an image array and visualize results."""
    # Convert numpy array to PIL Image.
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    font = ImageFont.load_default()

    # Preprocess image for model.
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

    # Make prediction.
    with torch.no_grad():
        predictions = model([image_tensor])[0]  # Get first (and only) prediction

    # Filter predictions by confidence.
    keep = predictions['scores'] > confidence_threshold
    boxes = predictions['boxes'][keep]
    labels = predictions['labels'][keep]
    scores = predictions['scores'][keep]

    # Draw each detection.
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        
        # Convert numeric label back to class name.
        class_name = LABEL_TO_CLASS.get(label.item(), "unknown")
        
        # Get corresponding color.
        box_color = LABEL_COLORS.get(class_name, "white")  # Default to white if unknown.
        
        # Draw bounding box.
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=2)
        
        # Prepare text.
        text = f"{class_name} {score:.2f}"
        
        # Calculate text size.
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw text background.
        draw.rectangle(
            [x1, y1 - text_height - 2, x1 + text_width, y1],
            fill=box_color
        )
        
        # Draw text.
        draw.text(
            (x1, y1 - text_height - 2),
            text,
            fill="black" if box_color in ["yellow"] else "white",  # Ensure good contrast
            font=font
        )

    return image_pil

```


```python
# Load model.
ckpt = results.checkpoint
with ckpt.as_directory() as ckpt_dir:
    model_path = os.path.join(ckpt_dir, "model.pt")
    model = models.detection.fasterrcnn_resnet50_fpn(num_classes=len(CLASS_TO_LABEL))
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

# URL for a test image.
url = "https://face-masks-data.s3.us-east-2.amazonaws.com/all/images/maksssksksss0.png"

# Load image from URL.
image_np = load_image_from_url(url)

# Run prediction and visualization.
result_image = predict_and_visualize(image_np, model, confidence_threshold=0.7)
result_image.show()
```

    Downloading: "https://download.pytorch.org/models/resnet50-0676ba61.pth" to /home/ray/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
    100%|██████████| 97.8M/97.8M [00:00<00:00, 221MB/s]



    
![png](README_files/README_28_1.png)
    


    [36m(autoscaler +34m19s)[0m [autoscaler] Downscaling node i-01100c929786011c2 (node IP: 10.0.3.85) due to node idle termination.
    [36m(autoscaler +34m19s)[0m [autoscaler] Cluster resized to {8 CPU, 2 GPU}.
    [36m(autoscaler +43m9s)[0m [autoscaler] Downscaling node i-0710850cb19aac7a5 (node IP: 10.0.21.161) due to node idle termination.
    [36m(autoscaler +43m9s)[0m [autoscaler] Downscaling node i-09dab0a72515cb0f4 (node IP: 10.0.49.121) due to node idle termination.
    [36m(autoscaler +1h48m59s)[0m [autoscaler] [1xT4:4CPU-16GB] Attempting to add 2 node(s) to the cluster (increasing from 0 to 2).
    [36m(autoscaler +1h48m59s)[0m [autoscaler] Launched 2 instances.
    [36m(autoscaler +1h49m59s)[0m [autoscaler] Cluster upscaled to {8 CPU, 2 GPU}.
    [36m(autoscaler +2h5m19s)[0m [autoscaler] Downscaling node i-02cef9eebb5e6d306 (node IP: 10.0.16.47) due to node idle termination.
    [36m(autoscaler +2h5m19s)[0m [autoscaler] Downscaling node i-0f1e07a8ce02f15b9 (node IP: 10.0.57.13) due to node idle termination.
    [36m(autoscaler +2h14m44s)[0m [autoscaler] [1xT4:4CPU-16GB] Attempting to add 2 node(s) to the cluster (increasing from 0 to 2).
    [36m(autoscaler +2h14m44s)[0m [autoscaler] Launched 2 instances.
    [36m(autoscaler +2h19m19s)[0m [autoscaler] Downscaling node i-045064ea2e35136ef (node IP: 10.0.32.202) due to node idle termination.
    [36m(autoscaler +2h19m19s)[0m [autoscaler] Cluster resized to {4 CPU, 1 GPU}.
    [36m(autoscaler +2h19m19s)[0m [autoscaler] Downscaling node i-059221026685f314a (node IP: 10.0.29.22) due to node idle termination.
    [36m(autoscaler +2h26m14s)[0m [autoscaler] [4CPU-8GB] Attempting to add 1 node(s) to the cluster (increasing from 0 to 1).
    [36m(autoscaler +2h26m14s)[0m [autoscaler] Launched 1 instances.
    [36m(autoscaler +2h29m19s)[0m [autoscaler] Downscaling node i-0b7270ab4e32550a7 (node IP: 10.0.31.120) due to node idle termination.
    [36m(autoscaler +2h44m29s)[0m [autoscaler] [1xT4:4CPU-16GB] Attempting to add 2 node(s) to the cluster (increasing from 0 to 2).
    [36m(autoscaler +2h44m29s)[0m [autoscaler] Launched 2 instances.
    [36m(autoscaler +2h45m29s)[0m [autoscaler] Cluster upscaled to {8 CPU, 2 GPU}.
    [36m(autoscaler +2h46m39s)[0m [autoscaler] Downscaling node i-05629e1b71e98266a (node IP: 10.0.59.199) due to node idle termination.
    [36m(autoscaler +2h46m39s)[0m [autoscaler] Downscaling node i-03ebbf25516d94d40 (node IP: 10.0.52.99) due to node idle termination.
    [36m(autoscaler +3h4m44s)[0m [autoscaler] [4CPU-8GB] Attempting to add 1 node(s) to the cluster (increasing from 0 to 1).
    [36m(autoscaler +3h4m49s)[0m [autoscaler] Launched 1 instances.


<div class="alert alert-block alert-warning"> <b> Note: You may notice that the results aren't optimal because we trained for only 2 epochs. 
Typically, training would require around 20 epochs.</b> 
<div>

### Store the trained model locally

After training, you can access the checkpoint, load the model weights, and save the model locally in your workspace. This allows you to easily download the model to your local machine, inspect the model, or do a sanity check. **Don't load the model and run batch inference directly from the workspace**, as this forcec the Ray cluster to copy the weights to other nodes, significantly slowing down the process. To enable faster batch inference, use Anyscale’s cluster storage to store the model instead.

```python
ckpt = results.checkpoint
with ckpt.as_directory() as ckpt_dir:
    model_path = os.path.join(ckpt_dir, "model.pt")
    model = models.detection.fasterrcnn_resnet50_fpn(num_classes=len(CLASS_TO_LABEL))
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

# Save the model locally.
save_path = "./saved_model/fasterrcnn_model_mask_detection.pth"  # Choose your path.
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if needed.
torch.save(model.state_dict(), save_path)
```

### Store the model on Anyscale Cluster Storage
You can store your model on Anyscale Cluster Storage (`/mnt/cluster_storage`) for faster batch inference or serving on Anyscale.  If your model needs to be accessed by multiple worker nodes in a distributed computing environment, storing it in cluster storage ensures all nodes load the model quickly and avoid redundant copies.

For more information, see: https://docs.anyscale.com/configuration/storage/


```python
ckpt = results.checkpoint
with ckpt.as_directory() as ckpt_dir:
    model_path = os.path.join(ckpt_dir, "model.pt")
    model = models.detection.fasterrcnn_resnet50_fpn(num_classes=len(CLASS_TO_LABEL))
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

# Save the model locally
save_path = "/mnt/cluster_storage/fasterrcnn_model_mask_detection.pth"  # Choose your path
os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if needed
torch.save(model.state_dict(), save_path)
```


### Store the model in the cloud
You can store your model in a cloud such as AWS S3, Google Cloud Storage, or Hugging Face. Store the model remotely on a cloud helps your team collaboration, versioning, and efficient deployment and inference. Later on, you can use `smart-open` to load the model from AWS S3, Google Cloud Storage, or use AutoModel to load the model from Hugging Face. See how to load the model from AWS S3 in the next notebook.

This sample code uploads your model to AWS S3. Be sure to install the boto3 library properly configure it with AWS credentials:

```python
import os
import torch
import boto3
import smart_open
from torchvision import models

# Define S3 details
S3_BUCKET = "your-s3-bucket-name"
S3_KEY = "path/in/s3/fasterrcnn_model_mask_detection.pth"
S3_URI = f"s3://{S3_BUCKET}/{S3_KEY}"

# Load the model checkpoint
ckpt = results.checkpoint
with ckpt.as_directory() as ckpt_dir:
    model_path = os.path.join(ckpt_dir, "model.pt")
    model = models.detection.fasterrcnn_resnet50_fpn(num_classes=len(CLASS_TO_LABEL))
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

# Upload to S3 directly using smart_open
try:
    with smart_open.open(S3_URI, "wb") as f:
        torch.save(model.state_dict(), f)
    print(f"Model successfully uploaded to {S3_URI}")
except Exception as e:
    print(f"Error uploading to S3: {e}")

```


## Clean up the cluster storage

You can see the files you stored in the cluster_storage. You can see that you created `/mnt/cluster_storage/face-mask-experiments_v1/` to store the training artifacts.


```python
!ls -lah /mnt/cluster_storage/
```

**Remember to clean up the cluster storage by removing it:**


```python
!rm -rf /mnt/cluster_storage/face-mask-experiments_v1/
```

## Next steps

For the following notebooks, **Anyscale has already uploaded a fine-tuned mask detection model (with a batch size of 20) to AWS S3**. The following notebook demonstrates how to download the model to an Anyscale cluster for batch inference, among other tasks.

However, feel free to use your own fine-tuned model (around 20 epochs) if you prefer.
