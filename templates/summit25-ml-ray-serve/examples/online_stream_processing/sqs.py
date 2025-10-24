import logging
from io import BytesIO
import asyncio
from typing import Dict

import boto3
import torch
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from ray import serve
from ray.serve.handle import DeploymentHandle, DeploymentResponse
from ray.serve.exceptions import BackPressureError
from ray._private.utils import get_or_create_event_loop


logger = logging.getLogger("ray.serve")

# Configure your AWS credentials and resource names
AWS_ACCESS_KEY_ID = "your-aws-access-key-id"
AWS_SECRET_ACCESS_KEY = "your-aws-secret-access-key"
AWS_SESSION_TOKEN = "your-aws-session-token"
S3_BUCKET_NAME = "your-s3-bucket-name"
SQS_QUEUE_NAME = "your-queue-name"


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "num_cpus": 1},
    max_queued_requests=3,  # Low queue capacity triggers backpressure early
    max_ongoing_requests=1,  # GPU-bound, process one at a time
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 10,
        "target_ongoing_requests": 1,  # Scale up when replica is busy
    },
)
class StableDiffusionXL:
    """Text-to-image model that generates images from text prompts."""

    def __init__(self):
        # Load model weights (download on first run)
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


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2})
class QueuePoller:
    """Continuously polls SQS queue and forwards messages to the model."""

    def __init__(self, model_handle: DeploymentHandle, queue_name: str):
        self.model_handle = model_handle
        # Track in-flight requests (prevents duplicate processing)
        self.processing_requests: Dict[str, asyncio.Task] = {}
        # Tracks consecutive backpressure failures for exponential backoff
        self.backpressure_counter: int = 0

        # Set up AWS clients
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            aws_session_token=AWS_SESSION_TOKEN,
        )
        sqs = session.resource("sqs", region_name="us-west-2")
        self.s3_client = session.client("s3")
        self.queue = sqs.get_queue_by_name(QueueName=queue_name)

        # Start background polling loop
        self.loop = get_or_create_event_loop()
        self.loop.create_task(self.run())

        # Event to shutdown polling
        self.shutdown_event = asyncio.Event()

    async def run(self):
        """Main polling loop: fetch messages from SQS and forward to model."""
        while True:
            if self.shutdown_event.is_set():
                logger.info("Breaking out of run loop because shutdown event is set")
                break

            # Fetch up to 10 messages (long polling reduces API costs)
            messages = await self.loop.run_in_executor(
                None,
                lambda: self.queue.receive_messages(
                    MaxNumberOfMessages=10,
                    WaitTimeSeconds=2,
                ),
            )
            for message in messages:
                # Skip if already processing (handles SQS visibility timeout redelivery)
                if message.message_id in self.processing_requests:
                    logger.info(f"Still processing {message.message_id}, skipping.")
                    continue

                logger.debug(f"Processing '{message.message_id}' from queue.")

                # Forward prompt to StableDiffusion (non-blocking)
                resp = self.model_handle.remote(message.body)
                assert isinstance(resp, DeploymentResponse)

                # Track request in background task
                new_task = self.loop.create_task(
                    self.process_finished_request(resp, message)
                )
                self.processing_requests[message.message_id] = new_task

            # Adaptive polling: slow down when backpressure detected
            # This gives autoscaler time to add replicas without overwhelming system
            if self.backpressure_counter == 0:
                await asyncio.sleep(0.1)  # Normal operation: poll frequently
            else:
                # Exponential backoff: 2s → 4s → 8s → 10s max
                backoff_time = min(10, 2**self.backpressure_counter)
                logger.info(
                    f"Model is overloaded, polling queue again after {backoff_time}s."
                )
                await asyncio.sleep(backoff_time)

    async def process_finished_request(self, resp: DeploymentResponse, queue_message):
        """Waits for model response and uploads result to S3."""
        try:
            # Wait for image generation to complete
            result = await resp

            # Offload blocking operations (PIL encoding + S3 upload) to thread pool
            filename = await self.loop.run_in_executor(
                None,
                self._upload_to_s3,
                result,
                queue_message.message_id,
            )

            logger.info(f"Uploaded {filename} to S3.")

        except BackPressureError as e:
            # Model queue full - don't delete message, let it retry
            self.backpressure_counter += 1
            logger.info(f"({queue_message.message_id}) {str(e)}")
        else:
            # Success - safe to delete from queue (at-least-once delivery)
            if self.backpressure_counter > 0:
                logger.info("Resetting backpressure counter")

            self.backpressure_counter = 0
            queue_message.delete()
            logger.info(f"Message {queue_message.message_id} deleted from queue.")
        finally:
            del self.processing_requests[queue_message.message_id]

    def _upload_to_s3(self, image, message_id: str) -> str:
        """Blocking helper: encode image and upload to S3."""
        # Convert PIL Image to bytes
        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        file_stream.seek(0)

        # Upload to S3
        filename = f"image_{message_id}.png"
        self.s3_client.upload_fileobj(
            Fileobj=file_stream,
            Bucket=S3_BUCKET_NAME,
            Key=filename,
            ExtraArgs={"ContentType": "image/png"},
        )
        return filename

    async def __del__(self):
        # Graceful shutdown: wait for in-flight requests to complete
        self.shutdown_event.set()
        while True:
            if len(self.processing_requests) == 0:
                break

            logger.info(
                f"Processing {len(self.processing_requests)} requests, waiting to "
                "shutdown."
            )
            await asyncio.sleep(2)


def app_builder() -> serve.Application:
    """Builds the application: QueuePoller → StableDiffusionXL."""
    return QueuePoller.bind(
        StableDiffusionXL.bind(),
        SQS_QUEUE_NAME,
    )
