import logging
from io import BytesIO
import asyncio
from typing import Dict
from collections import defaultdict

import boto3
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse
from ray.serve.exceptions import BackPressureError
from ray._private.utils import get_or_create_event_loop
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm import SamplingParams


# Set up the Ray Serve logger.
logger = logging.getLogger("ray.serve")

AWS_ACCESS_KEY_ID = "your-aws-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-aws-secret-access-key"
AWS_SESSION_TOKEN = "your-aws-session-token"
S3_BUCKET_NAME = "your-s3-bucket-name"
SQS_QUEUE_1 = "your-first-queue-name"
SQS_QUEUE_2 = "your-second-queue-name"


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 1},
    max_queued_requests=3,
    max_ongoing_requests=1,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 1,
    },
)
class StableDiffusionXL:
    def __init__(self):
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"

        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.pipe.to("cuda")

    def __call__(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"
        logger.info("Prompt: [%s]", prompt)
        image = self.pipe(prompt, height=img_size, width=img_size).images[0]

        return image


@serve.deployment(
    max_queued_requests=40,
    max_ongoing_requests=40,
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 20,
        "upscaling_factor": 0.2,
    },
)
class VLLMDeployment:
    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.request_id = 0
    
    async def __call__(self, prompt):
        results_generator = self.engine.generate(
            prompt=(
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful assistant.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            request_id=str(self.request_id),
            sampling_params=SamplingParams(temperature=0.01, max_tokens=1000),
        )
        self.request_id += 1

        final_output = None
        async for request_output in results_generator:
            final_output = request_output.outputs[0].text
        return final_output


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.2})
class QueuePoller:
    def __init__(self, handles: Dict[str, DeploymentHandle], queues: Dict[str, str]):
        # Map from model name -> deployment handles
        self.handles = handles
        # Ongoing requests that are executing at a downstream ML model
        self.processing_requests: Dict[str, asyncio.Task] = {}
        # A map from model name -> counter that tracks, for each model,
        # how many consecutive times forwarding a request has failed
        # with backpressure error.
        # NOTE: a request fails with backpressure error when the number
        # of queued requests exceeds `max_queued_requests` for that
        # deployment.
        self.backpressure_counters: Dict[str, int] = defaultdict(int)

        self.shutdown_event = asyncio.Event()

        # Set up S3 clients
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN
        )
        sqs = session.resource("sqs", region_name="us-west-2")
        self.s3_client = session.client("s3")
        self.queues = {
            model: sqs.get_queue_by_name(QueueName=queue_name)
            for model, queue_name in queues.items()
        }

        # Run loops in background to continuously poll SQS queue
        self.loop = get_or_create_event_loop()
        for model_name in self.handles:
            self.loop.create_task(self.run(model_name))

    async def process_finished_request(
        self, resp: DeploymentResponse, model_name: str, queue_message
    ):
        try:
            result = await resp

            # Upload result to s3
            if model_name == "stable_diffusion":
                # Save image to bytes buffer
                file_stream = BytesIO()
                result.save(file_stream, "PNG")
                file_stream.seek(0)

                filename = f"image_{queue_message.message_id}.png"
                self.s3_client.upload_fileobj(
                    Fileobj=file_stream, 
                    Bucket=S3_BUCKET_NAME,
                    Key=filename,
                    ExtraArgs={"ContentType":"image/png"}
                )
            else:
                assert model_name == "llm"
                # Save output to bytes buffer
                file_stream = BytesIO(result.encode("utf-8"))

                filename = f"response_{queue_message.message_id}.txt"
                self.s3_client.upload_fileobj(
                    Fileobj=file_stream, 
                    Bucket=S3_BUCKET_NAME,
                    Key=filename,
                )

            logger.info(f"Uploaded {filename} to S3.")

        except BackPressureError as e:
            self.backpressure_counters[model_name] += 1
            logger.info(f"({queue_message.message_id}) {str(e)}")
        else:
            if self.backpressure_counters[model_name] > 0:
                logger.info(f"Resetting counter for {model_name}")

            self.backpressure_counters[model_name] = 0
            queue_message.delete()
            logger.info(f"Message {queue_message.message_id} deleted from queue.")
        finally:
            del self.processing_requests[queue_message.message_id]

    async def run(self, model_name: str):
        while True:
            if self.shutdown_event.is_set():
                logger.info("Breaking out of run loop because shutdown event is set")
                break

            messages = await self.loop.run_in_executor(
                None,
                lambda: self.queues[model_name].receive_messages(
                    MaxNumberOfMessages=10, WaitTimeSeconds=2,
                )
            )
            for message in messages:
                if message.message_id in self.processing_requests:
                    # We could re-receive the same message if the time for which the
                    # message has been queued in the deployment handle has exceeded
                    # the SQS queue's visibility timeout.
                    # Although there's no guarantee that another QueuePoller replica
                    # won't pick up this message, we can ensure at-least-once execution
                    logger.info(f"Still processing {message.message_id}, skipping.")
                    continue

                logger.debug(f"Processing '{message.message_id}' from queue.")

                # Forward request to downstream models: StableDiffisionXL or VLLMDeployment
                resp = self.handles[model_name].remote(message.body)
                assert isinstance(resp, DeploymentResponse)

                new_task = self.loop.create_task(
                    self.process_finished_request(resp, model_name, message)
                )
                self.processing_requests[message.message_id] = new_task

            if self.backpressure_counters[model_name] == 0:
                await asyncio.sleep(0.1)
            else:
                backoff_time = min(10, 2 ** self.backpressure_counters[model_name])
                logger.info(
                    f"'{model_name}' models are overloaded, polling queue again after "
                    f"{backoff_time}s."
                )
                await asyncio.sleep(backoff_time)

    async def __del__(self):
        self.shutdown_event.set()
        # Wait until all ongoing requests are finished processing.
        while True:
            if len(self.processing_requests) == 0:
                break

            logger.info(
                f"Processing {len(self.processing_requests)} requests, waiting to "
                "shutdown."
            )
            await asyncio.sleep(2)


# -- Application entrypoint (to be imported) ---
def app_builder(cli_args: Dict[str, str]) -> serve.Application:
    parser = make_arg_parser()
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f'--{key}', str(value)])
    parsed_args = parser.parse_args(args=arg_strings)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    pg_resources = [{"CPU": 1.0}]
    for i in range(engine_args.tensor_parallel_size):
        pg_resources.append({"CPU": 2.0, "GPU": 1.0}) # for the vLLM workers

    return QueuePoller.bind(
        {
            "stable_diffusion": StableDiffusionXL.bind(),
            "llm": VLLMDeployment.options(
                placement_group_bundles=pg_resources,
                placement_group_strategy="STRICT_PACK"
            ).bind(engine_args)
        },
        {
            "stable_diffusion": SQS_QUEUE_1,
            "llm": SQS_QUEUE_2,
        },
    )