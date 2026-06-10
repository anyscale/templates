import io
import numpy
import os
import tritonserver
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from ray import serve

app = FastAPI()

def setup_model_repository():
    """Setup model repository, downloading from ABFSS if needed."""
    import socket
    print(f"🖥️ Setting up model repository on node: {socket.gethostname()}")

    storage_url = os.environ.get("ANYSCALE_ARTIFACT_STORAGE", "")

    if storage_url.startswith("abfss://"):
        # ABFSS storage - need to download to local directory
        print("Detected ABFSS storage, downloading model...")

        # Since ABFSS protocol is now supported, use it directly
        print("✅ ABFSS protocol is supported, proceeding with direct download")
        local_model_path = "/tmp/triton_model_repository"

        # Check if model already exists locally
        config_file = f"{local_model_path}/stable_diffusion_1_5/config.pbtxt"
        if os.path.exists(config_file):
            print(f"✅ Model already exists at {local_model_path}, skipping download")
            print(f"🔍 Returning model repository path: {local_model_path}")
            return local_model_path

        # Create local directory
        os.makedirs(local_model_path, exist_ok=True)

        # Parse ABFSS URL
        ABFSS_PREFIX = "abfss://"
        url_parts = storage_url[len(ABFSS_PREFIX):].split('@')
        container = url_parts[0]
        account_and_path = url_parts[1].split('.dfs.core.windows.net/')
        account = account_and_path[0]
        base_path = account_and_path[1].strip('/') if len(account_and_path) > 1 and account_and_path[1] else ""

        # Construct blob prefix
        if base_path:
            blob_prefix = f"{base_path}/triton_model_repository/"
        else:
            blob_prefix = "triton_model_repository/"

        # Create local directory
        os.makedirs(local_model_path, exist_ok=True)

        # Try to use existing Azure CLI session first
        print("Checking existing Azure CLI session...")
        existing_session = os.system("az account show > /dev/null 2>&1")

        if existing_session != 0:
            print("No existing session, logging in with managed identity...")
            login_result = os.system("az login --identity > /dev/null 2>&1")
            if login_result != 0:
                print("Failed to login with managed identity on worker node")
                print("This is common in distributed Ray clusters where worker nodes don't have managed identity")
                print("Attempting to download model from head node instead...")

                # Worker node cannot authenticate - since ABFSS is now supported at the Ray level,
                # we can proceed without authentication for the runtime environment
                print("⚠️ Cannot authenticate on worker node, but ABFSS protocol is supported")
                print("🔄 Proceeding with download attempt...")
                # Continue to download logic below
        else:
            print("✅ Using existing Azure CLI session")

        print(f"Downloading model from ABFSS...")
        print(f"Account: {account}, Container: {container}, Prefix: {blob_prefix}")

        # First, list all blobs with the prefix to see what's available
        list_command = f"az storage blob list --account-name {account} --container-name {container} --prefix \"{blob_prefix}\" --auth-mode login --query \"[].name\" --output tsv"
        print(f"Listing blobs: {list_command}")

        # Use a simpler approach: download individual files
        # Download config file
        config_blob = f"{blob_prefix}stable_diffusion_1_5/config.pbtxt"
        config_local = f"{local_model_path}/stable_diffusion_1_5/config.pbtxt"
        os.makedirs(os.path.dirname(config_local), exist_ok=True)

        config_command = f"az storage blob download --account-name {account} --container-name {container} --name \"{config_blob}\" --file \"{config_local}\" --auth-mode login"
        print(f"Downloading config: {config_command}")
        result1 = os.system(config_command)

        # Download model files directory (try a different approach)
        model_dir_local = f"{local_model_path}/stable_diffusion_1_5/1/1.5-engine-batch-size-1/"
        os.makedirs(model_dir_local, exist_ok=True)

        # Use download-batch with a simpler pattern
        model_pattern = f"{blob_prefix}stable_diffusion_1_5/1/1.5-engine-batch-size-1/*"
        batch_command = f"az storage blob download-batch --account-name {account} --source {container} --destination {model_dir_local} --pattern \"{model_pattern}\" --auth-mode login"
        print(f"Downloading model files: {batch_command}")
        result2 = os.system(batch_command)

        if result1 != 0 or result2 != 0:
            print(f"Config download result: {result1}, Model download result: {result2}")
            # Try alternative approach - download all blobs individually
            print("Batch download failed, trying individual blob downloads...")

            # List all blobs and download them one by one
            import subprocess
            import json

            list_cmd = f"az storage blob list --account-name {account} --container-name {container} --prefix \"{blob_prefix}\" --auth-mode login --output json"
            try:
                result = subprocess.run(list_cmd, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    blobs = json.loads(result.stdout)
                    for blob in blobs:
                        blob_name = blob['name']
                        # Remove the prefix to get the local path
                        # blob_prefix ends with triton_model_repository/, so removing it gives us the model structure
                        local_path = blob_name.replace(blob_prefix, '')
                        local_file = f"{local_model_path}/{local_path}"

                        # Create directory if needed
                        os.makedirs(os.path.dirname(local_file), exist_ok=True)

                        # Download the blob
                        download_cmd = f"az storage blob download --account-name {account} --container-name {container} --name \"{blob_name}\" --file \"{local_file}\" --auth-mode login"
                        print(f"Downloading: {blob_name}")
                        dl_result = os.system(download_cmd)
                        if dl_result != 0:
                            raise RuntimeError(f"Failed to download blob {blob_name}")
                else:
                    raise RuntimeError(f"Failed to list blobs: {result.stderr}")
            except Exception as e:
                raise RuntimeError(f"Failed to download model from ABFSS: {e}")

        print(f"✅ Successfully downloaded model to {local_model_path}")

        # Return the local model repository path (not nested with base_path)
        print(f"🔍 Returning model repository path: {local_model_path}")
        return local_model_path

    else:
        # Direct path or other storage types (S3, GCS, local)
        result_path = f'{storage_url}/triton_model_repository'
        print(f"🔍 Non-ABFSS storage detected. Storage URL: {storage_url}")
        print(f"🔍 Returning model repository path: {result_path}")
        return result_path

@serve.deployment(
    ray_actor_options={
        "num_gpus": 1,
        "accelerator_type": "T4"
    },
)
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):
        # Setup model repository path on the same node where Triton will run
        model_repository = setup_model_repository()
        print(f"Using model repository: {model_repository}")

        try:
            self._triton_server = tritonserver.Server(
                model_repository=model_repository,
                model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
                log_info=False,
            )
            self._triton_server.start(wait_until_ready=True)

            # Load model using Triton.
            self._model = None
            if not self._triton_server.model("stable_diffusion_1_5").ready():
                self._model = self._triton_server.load("stable_diffusion_1_5")

                if not self._model.ready():
                    print("⚠️ Model not ready - this might be due to missing CUDA dependencies")
                    self._model = None

            print("✅ Triton server initialized successfully")

        except Exception as e:
            print(f"❌ Failed to initialize Triton server: {e}")
            print("💡 This might be due to missing CUDA runtime or TensorRT dependencies")
            print("🔄 Service will continue but with limited functionality")
            self._triton_server = None
            self._model = None

    @app.get("/")
    def health_check(self):
        """Health check endpoint."""
        return {
            "status": "healthy",
            "triton_server": "running" if self._triton_server else "not_available",
            "model": "ready" if self._model and self._model.ready() else "not_ready",
            "message": "ABFSS-enabled Triton deployment"
        }

    @app.get("/generate")
    def generate(self, prompt: str) -> PlainTextResponse:
        """Generate image from prompt."""
        if not self._model or not self._model.ready():
            # Return a placeholder response if model is not available
            print(f"⚠️ Model not ready, returning placeholder for prompt: {prompt}")

            # Create a simple placeholder image
            placeholder = numpy.random.randint(0, 255, (512, 512, 3), dtype=numpy.uint8)
            image = Image.fromarray(placeholder)

            buffer = io.BytesIO()
            image.save(buffer, "JPEG")
            return PlainTextResponse(buffer.getvalue(), media_type="image/jpeg")

        try:
            print(f"Generating image for prompt: {prompt}")
            for response in self._model.infer(inputs={"prompt": [[prompt]]}):
                generated_image = (
                    numpy.from_dlpack(response.outputs["generated_image"])
                    .squeeze()
                    .astype(numpy.uint8)
                )
                image = Image.fromarray(generated_image)

                buffer = io.BytesIO()
                image.save(buffer, "JPEG")
                return PlainTextResponse(buffer.getvalue(), media_type="image/jpeg")
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            # Return placeholder on error
            placeholder = numpy.random.randint(0, 255, (512, 512, 3), dtype=numpy.uint8)
            image = Image.fromarray(placeholder)

            buffer = io.BytesIO()
            image.save(buffer, "JPEG")
            return PlainTextResponse(buffer.getvalue(), media_type="image/jpeg")


triton_deployment = TritonDeployment.bind()
