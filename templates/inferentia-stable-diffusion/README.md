# Run stable diffusion XL on inferentia

This template provides an example of compiling and serving the [stable diffusion XL model](https://huggingface.co/docs/diffusers/using-diffusers/sdxl) using Ray Serve on inferentia. It uses code from the [AWS example repo](https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_sdxl_base_and_refiner_1024_inference.ipynb). You can find examples for other models in the aws-neuron repo.

| Template Specification | Description |
| ---------------------- | ----------- |
| Time to Run | Around 1 hour to compile the model, ~5 mins to load and run the model. |
| Minimum Compute Requirements | The default is a head node with one inferentia accelerator (2 neuron cores).|
| Cluster Environment | This template uses a pre-built image. You can build a new image yourself by running the contents of setup.sh as post-build commands in a new cluster environment.  |

## Compiling the model

After launching this template, within the Workspace terminal you can run following command to compile the stable diffusion model.

`python compile.py`

* Note - To avoid the workspace snapshotting failure due to the large files, run `git init` and add the output folder to .gitignore.

## Running the model

To launch the serve application:

`cd serve`

`serve run sd_serve:sd_app`

Once the serve application has started successfully, open another terminal and test it using. The output is saved in `image.png`:

`python query.py`


## Roll out as Anyscale service

This part is not covered in the template, but can be easily added on. In order to deploy to an Anyscale Service, you would first need to upload the compiled model files to an artifact storage, like S3, or add them into a docker image, since runtime environments cannot be larger than 500MB. Then, update the init method in the serve code to fetch the model from the right location. After those changes, run:

`anyscale service rollout -f srvc.yaml`
