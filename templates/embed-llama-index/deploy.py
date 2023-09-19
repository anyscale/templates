from llama_index import (
    load_index_from_storage, 
    ServiceContext, 
    StorageContext, 
    LangchainEmbedding,
)

import os
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles

from ray import serve
from starlette.responses import FileResponse 
from sse_starlette.sse import EventSourceResponse

import os

from conf import RAY_BLOGS_INDEX, RAY_DOCS_INDEX
from dotenv import load_dotenv

load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise RuntimeError("Please add the OPENAI_API_KEY environment variable to run this script. Run the following in your terminal `export OPENAI_API_KEY=...`")

openai_api_key = os.environ["OPENAI_API_KEY"]


from fastapi import FastAPI
from starlette.responses import StreamingResponse

import asyncio
from queue import Empty

from starlette.responses import StreamingResponse
import logging


app = FastAPI()
logger = logging.getLogger("ray.serve")

@serve.deployment(route_prefix="/")
@serve.ingress(app)
class Askmeanything:
    def __init__(self):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        # Define the embedding model used to embed the query.
        query_embed_model = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
        service_context = ServiceContext.from_defaults(embed_model=query_embed_model)

        # Load the vector stores that were created earlier.
        storage_context = StorageContext.from_defaults(persist_dir=RAY_DOCS_INDEX)
        ray_docs_index = load_index_from_storage(storage_context, service_context=service_context)   

        storage_context = StorageContext.from_defaults(persist_dir=RAY_BLOGS_INDEX)
        ray_blogs_index = load_index_from_storage(storage_context, service_context=service_context)  

        # Define 2 query engines:
        #   1. Ray documentation
        #   2. Anyscale blogs
        self.ray_docs_engine = ray_docs_index.as_query_engine(similarity_top_k=5, service_context=service_context, streaming=True)
        self.ray_blogs_engine = ray_blogs_index.as_query_engine(similarity_top_k=5, service_context=service_context, streaming=True)


    @app.get("/")
    async def read_index(self):
        return FileResponse('./static/index.html')

    @app.get("/event")
    async def event(self, query: str, request: Request, engine: str = "blogs"):
        streaming_response = self.ray_docs_engine.query(query, )
        if engine == "docs":
            streaming_response = self.ray_docs_engine.query(query, )
        elif engine == "blogs":
            streaming_response = self.ray_blogs_engine.query(query)
        async def event_generator():
            for text in streaming_response.response_gen:
                if await request.is_disconnected():
                    break

                # Checks for new messages and return them to client if any
                if text:
                    yield {
                        "event": "stream",
                        "data": text,
                    }

                await asyncio.sleep(0.1)
            yield {
                "event": "stream",
                "data": "<<end>>"
            }

        return EventSourceResponse(event_generator())

    async def consume_streamer(self, streamer):
        while True:
            try:
                async for token in streamer:
                    logger.info(f'Yielding token: "{token}"')
                    yield token
                break
            except Empty:
                # The streamer raises an Empty exception if the next token
                # hasn't been generated yet. `await` here to yield control
                # back to the event loop so other coroutines can run.
                await asyncio.sleep(0.001)

    @app.post("/query")
    def query(self, query: str, engine: str = "blogs"):
        # Route the query to the appropriate engine.
        if engine == "docs":
            streaming_response = self.ray_docs_engine.query(query, )
            return self.consume_streamer(streaming_response.response_gen)
        elif engine == "blogs":
            streaming_response = self.ray_blogs_engine.query(query)
            return self.consume_streamer(streaming_response.response_gen)

    @app.get("/ask")
    async def ask(self, query:str, engine_to_use: str = "blogs"):
        return StreamingResponse(
            self.query(engine_to_use, query), media_type="text/plain"
        )

def deploy():
    serve.run(Askmeanything.bind())

if __name__ == '__main__':
    deploy()

    
