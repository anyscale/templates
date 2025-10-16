from typing import List
from fastapi import FastAPI, HTTPException
import ray
from ray import serve
from ray.serve.handle import DeploymentHandle

fastapi_app = FastAPI()

@serve.deployment
@serve.ingress(fastapi_app)
class Ingress:
    """Main ingress deployment that orchestrates the video processing pipeline.
    
    This deployment manages the streaming pipeline for video processing:
    1. Decodes video into frames
    2. Processes frames through face detection and encoding
    3. Concatenates processed frames back into video
    
    Args:
        decode_video: Deployment that handles video decoding
        fused_detect_encode: Deployment that handles face detection and frame encoding
        concat_video: Deployment that handles video concatenation
    """
    
    def __init__(
        self,
        decode_video: DeploymentHandle,
        fused_detect_encode: DeploymentHandle,
        concat_video: DeploymentHandle,
    ) -> None:
        # Initialize with local routing preference for better performance
        decode_video._init(_prefer_local_routing=True)
        self.decode_video = decode_video.options(stream=True)
        self.fused_detect_encode = fused_detect_encode
        self.concat_video = concat_video

    @fastapi_app.post("/detect_faces")
    async def detect_faces(
        self,
        video_url: str,
        output_path: str,
        decode_batch_size: int = 20,  # Number of frames to process in each batch
    ) -> str:
        """Process video through the face detection pipeline.
        
        Args:
            video_url: URL or path to input video
            output_path: Path where processed video will be saved
            decode_batch_size: Number of frames to decode in each batch
            
        Returns:
            str: Path to the processed video
            
        Raises:
            HTTPException: If video processing fails
        """
        try:
            # Start streaming frames from the video decoder
            frames_iter = self.decode_video.decode.remote(video_url, decode_batch_size)
            
            # Process frames as they become available
            encoded_video_refs: List[ray.ObjectRef] = []
            obj_ref_gen = await frames_iter._to_object_ref_gen()
            
            async for frame_ref in obj_ref_gen:
                # Process each frame through face detection and encoding
                encoded_video_refs.append(
                    self.fused_detect_encode.run.remote(frame_ref)
                )
            
            # Concatenate all processed frames into final video
            result = await self.concat_video.concat.remote(
                output_path, 
                *encoded_video_refs
            )
            
            return result
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Video processing failed: {str(e)}"
            ) 