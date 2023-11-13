# Run Llama2 on inferentia

This template provides an example of serving the [compiled Llama 2 model](https://huggingface.co/blog/inferentia-llama2) using Ray Serve on inferentia. We use the Llama2 7B - B (budget) model in this example. 

| Template Specification | Description |
| ---------------------- | ----------- |
| Time to Run | Around 5 mins. |
| Minimum Compute Requirements | The default is a head node with 2 inferentia cores (1 accelerator).|
| Cluster Environment | This template uses a pre-built image. You can build a new image yourself by running the contents of setup.sh as post-build commands in a new cluster environment.  |

You can use any of the other models from the blog - Since the others require 24 neuron cores, you would need to update the `ray_actor_options` in the serve deployment configurations.

## Running Ray Serve

To launch the serve application:

`serve run serve.yaml`

Once the serve application has started successfully, open another terminal and test it using. The output is saved in `image.png`:

`python query.py`


## Roll out as Anyscale service

This part is not covered in the template, but can be easily added on. In order to deploy to an Anyscale Service, you would first need to upload the compiled model files to an artifact storage, like S3, or add them into a docker image, since runtime environments cannot be larger than 500MB. Then, update the init method in the serve code to fetch the model from the right location. After those changes, run:

`anyscale service rollout -f srvc.yaml`

