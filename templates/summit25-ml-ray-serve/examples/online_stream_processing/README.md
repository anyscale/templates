# SQS Queue Processing with Ray Serve

This example demonstrates queue-based ML inference using Ray Serve, SQS, and Stable Diffusion XL for text-to-image generation.

## Key Concepts

- Understanding the queue polling pattern with async/await
- Learning how backpressure works with exponential backoff
- Seeing GPU-based model autoscaling in action
- Building a reliable at-least-once delivery system
- Offloading blocking I/O to thread pools

## Architecture

```
SQS Queue ──> QueuePoller (1 replica) ──> StableDiffusionXL (autoscaling) ──> S3
```

Data flow:
1. QueuePoller continuously polls the SQS queue for messages
2. Messages (text prompts) are forwarded to StableDiffusionXL
3. Generated images are uploaded to S3 (via thread pool executor)
4. Successfully processed messages are deleted from the queue

## Step-by-Step Walkthrough

### 1. Initialization

```python
@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.2})
class QueuePoller:
    def __init__(self, model_handle: DeploymentHandle, queue_name: str):
        self.model_handle = model_handle
        # Track in-flight requests (prevents duplicate processing)
        self.processing_requests: Dict[str, asyncio.Task] = {}
        # Tracks consecutive backpressure failures for exponential backoff
        self.backpressure_counter: int = 0
        
        # Set up AWS clients
        session = boto3.Session(...)
        self.s3_client = session.client("s3")
        self.queue = sqs.get_queue_by_name(QueueName=queue_name)
        
        # Start background polling loop
        self.loop.create_task(self.run())
```

The poller starts with:
- A handle to the StableDiffusionXL deployment
- An empty dictionary to track in-flight requests (prevents duplicate processing)
- A backpressure counter for exponential backoff
- AWS SQS and S3 clients
- A background task that runs the polling loop

### 2. Polling Loop

```python
async def run(self):
    while True:
        messages = await self.loop.run_in_executor(
            None,
            lambda: self.queue.receive_messages(
                MaxNumberOfMessages=10, WaitTimeSeconds=2,
            )
        )
```

The loop:
- Runs continuously until shutdown
- Retrieves up to 10 messages per poll
- Uses long polling (`WaitTimeSeconds=2`) to reduce API costs
- Runs SQS calls in a thread pool to avoid blocking the async loop

### 3. Message Processing

```python
for message in messages:
    if message.message_id in self.processing_requests:
        logger.info(f"Still processing {message.message_id}, skipping.")
        continue

    resp = self.model_handle.remote(message.body)
    new_task = self.loop.create_task(
        self.process_finished_request(resp, message)
    )
    self.processing_requests[message.message_id] = new_task
```

For each message:
1. Check if already processing (handles SQS visibility timeout redelivery)
2. Forward the prompt to StableDiffusionXL
3. Create an async task to handle the response
4. Track the task in `processing_requests`

### 4. Result Handling

```python
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
```

Blocking upload helper used above:

```python
def _upload_to_s3(self, image, message_id: str) -> str:
    """Blocking helper: encode image and upload to S3."""
    file_stream = BytesIO()
    image.save(file_stream, "PNG")
    file_stream.seek(0)

    filename = f"image_{message_id}.png"
    self.s3_client.upload_fileobj(
        Fileobj=file_stream,
        Bucket=S3_BUCKET_NAME,
        Key=filename,
        ExtraArgs={"ContentType": "image/png"},
    )
    return filename
```

When processing completes:
1. Wait for the model response
2. Convert the PIL Image to a byte stream
3. Upload to S3 with the message ID as the filename
4. This provides idempotency—same message ID always produces same filename

## Backpressure Handling

### What Triggers Backpressure?

StableDiffusionXL has limited queue capacity:

```python
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
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        self.pipe.to("cuda")

    def __call__(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"
        image = self.pipe(prompt, height=img_size, width=img_size).images[0]
        return image
```

When more than 3 requests are queued, Ray Serve raises `BackPressureError`.

### Backpressure Response

```python
except BackPressureError as e:
    self.backpressure_counter += 1
    logger.info(f"({queue_message.message_id}) {str(e)}")
else:
    if self.backpressure_counter > 0:
        logger.info("Resetting backpressure counter")
    self.backpressure_counter = 0
    queue_message.delete()
finally:
    del self.processing_requests[queue_message.message_id]
```

On backpressure:
- Increment the counter
- Do NOT delete the message (it stays in SQS for retry)
- Remove from local tracking

On success:
- Reset counter to 0
- Delete the message from SQS

### Exponential Backoff

This logic lives at the end of `QueuePoller.run(...)` (the polling loop) and adapts polling frequency based on recent backpressure events.

```python
class QueuePoller:
    async def run(self):
        # ... poll SQS, forward messages, track tasks ...

        # Adaptive polling: slow down when backpressure is detected
        if self.backpressure_counter == 0:
            await asyncio.sleep(0.1)
        else:
            backoff_time = min(10, 2**self.backpressure_counter)
            logger.info(
                f"Model is overloaded, polling queue again after {backoff_time}s."
            )
            await asyncio.sleep(backoff_time)
```

The polling frequency adapts based on backpressure:

| Counter | Backoff Time | State |
|---------|--------------|-------|
| 0 | 0.1s | Normal operation |
| 1 | 2s | First backpressure detected |
| 2 | 4s | Still overloaded |
| 3 | 8s | Continued overload |
| 4+ | 10s (capped) | Severely overloaded |

This gives the autoscaler time to add replicas without overwhelming the system.

## Autoscaling Flow Example

Let's trace what happens when the queue suddenly gets 50 messages:

**T=0s:** Queue has 50 messages, 1 StableDiffusion replica exists
- Poller retrieves 10 messages
- Forwards all 10 to StableDiffusion
- 3 queue in the deployment, 1 actively processing
- Remaining 6 fail with `BackPressureError`

**T=0s:** Backpressure counter = 1
- Poller sleeps for 2 seconds
- Messages stay in SQS (not deleted)
- Ray Serve detects high queue depth

**T=2s:** Poller wakes up, retrieves messages again
- Still overloaded, counter = 2
- Sleeps for 4 seconds
- Ray Serve starts scaling up (adds replicas)

**T=6s:** Poller wakes up, 3 replicas now exist
- More capacity available, some requests succeed
- Mix of successes and backpressure
- Counter stays elevated but stabilizes

**T=10s+:** System reaches steady state
- Enough replicas to handle load
- Requests consistently succeed
- Counter resets to 0, normal polling resumes
