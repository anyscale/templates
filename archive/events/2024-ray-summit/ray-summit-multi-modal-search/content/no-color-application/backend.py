import asyncio
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from ray.serve import deployment, ingress
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

def vector_search(
    vector_search_index_name: str,
    vector_search_path: str,
    embedding: list[float],
    n: int,
    min_price: int,
    max_price: int,
    min_rating: float,
    categories: list[str],
    colors: list[str],
    seasons: list[str],
    cosine_score_threshold: float = 0.92,
) -> list[dict]:
    return [
        {
            "$vectorSearch": {
                "index": vector_search_index_name,
                "path": vector_search_path,
                "queryVector": embedding.tolist(),
                "numCandidates": 100,
                "limit": n,
                "filter": {
                    "price": {"$gte": min_price, "$lte": max_price},
                    "rating": {"$gte": min_rating},
                    "category": {"$in": categories},
#                    "color": {"$in": colors},
                    "season": {"$in": seasons},
                },
            }
        },
        {
            "$project": {
                "_id": 1,
                "img": 1,
                "name": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
        {"$match": {"score": {"$gte": cosine_score_threshold}}},
    ]


def lexical_search(text_search: str, text_search_index_name: str) -> list[dict]:
    return [
        {
            "$search": {
                "index": text_search_index_name,
                "text": {
                    "query": text_search,
                    "path": "name",
                },
            }
        }
    ]


def match_on_metadata(
    min_price: int,
    max_price: int,
    min_rating: float,
    n: int,
    categories: list[str] | None = None,
    colors: list[str] | None = None,
    seasons: list[str] | None = None,
) -> list[dict]:
    match_spec = {
        "price": {
            "$gte": min_price,
            "$lte": max_price,
        },
        "rating": {"$gte": min_rating},
    }
    if categories:
        match_spec["category"] = {"$in": categories}
#    if colors:
#        match_spec["color"] = {"$in": colors}
    if seasons:
        match_spec["season"] = {"$in": seasons}

    return [
        {
            "$match": match_spec,
        },
        {"$limit": n},
    ]


def convert_rank_to_score(score_name: str, score_penalty: float) -> list[dict]:
    return [
        {
            "$group": {
                "_id": None,
                "docs": {
                    "$push": "$$ROOT",
                },
            }
        },
        {
            "$unwind": {
                "path": "$docs",
                "includeArrayIndex": "rank",
            }
        },
        {
            "$addFields": {
                score_name: {
                    "$divide": [
                        1.0,
                        {"$add": ["$rank", score_penalty, 1]},
                    ]
                }
            }
        },
        {
            "$project": {
                score_name: 1,
                "_id": "$docs._id",
                "name": "$docs.name",
                "img": "$docs.img",
            }
        },
    ]


def rerank_by_combined_score(
    vs_score_name: str, fts_score_name: str, n: int
) -> list[dict]:
    return [
        {
            "$group": {
                "_id": "$name",
                "img": {"$first": "$img"},
                vs_score_name: {"$max": f"${vs_score_name}"},
                fts_score_name: {"$max": f"${fts_score_name}"},
            }
        },
        {
            "$project": {
                "_id": 1,
                "img": 1,
                vs_score_name: {"$ifNull": [f"${vs_score_name}", 0]},
                fts_score_name: {"$ifNull": [f"${fts_score_name}", 0]},
            }
        },
        {
            "$project": {
                "name": "$_id",
                "img": 1,
                vs_score_name: 1,
                fts_score_name: 1,
                "score": {"$add": [f"${fts_score_name}", f"${vs_score_name}"]},
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": n},
    ]


def hybrid_search(
    collection_name: str,
    text_search: str,
    text_search_index_name: str,
    vector_search_index_name: str,
    vector_search_path: str,
    embedding: list[float],
    n: int,
    min_price: int,
    max_price: int,
    min_rating: float,
    categories: list[str],
    colors: list[str],
    seasons: list[str],
    vector_penalty: int,
    full_text_penalty: int,
    cosine_score_threshold: float = 0.92,
) -> list[dict]:
    # 1. Perform vector search
    vector_search_stages = vector_search(
        vector_search_index_name=vector_search_index_name,
        vector_search_path=vector_search_path,
        embedding=embedding,
        n=n,
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        categories=categories,
        colors=colors,
        seasons=seasons,
        cosine_score_threshold=cosine_score_threshold,
    )
    convert_vector_rank_to_score_stages = convert_rank_to_score(
        score_name="vs_score", score_penalty=vector_penalty
    )

    # 2. Perform lexical search
    lexical_search_stages = lexical_search(text_search=text_search, text_search_index_name=text_search_index_name)
    post_filter_stages = match_on_metadata(
        min_price=min_price,
        max_price=max_price,
        min_rating=min_rating,
        categories=categories,
        colors=colors,
        seasons=seasons,
        n=n,
    )
    convert_text_rank_to_score_stages = convert_rank_to_score(
        score_name="fts_score", score_penalty=full_text_penalty
    )

    # 3. Rerank by combined score
    rerank_stages = rerank_by_combined_score(
        vs_score_name="vs_score", fts_score_name="fts_score", n=n
    )

    # 4. Put it all together
    return [
        *vector_search_stages,
        *convert_vector_rank_to_score_stages,
        {
            "$unionWith": {
                "coll": collection_name,
                "pipeline": [
                    *lexical_search_stages,
                    *post_filter_stages,
                    *convert_text_rank_to_score_stages,
                ],
            }
        },
        *rerank_stages,
    ]


@deployment
class EmbeddingModel:
    def __init__(self, model: str = "thenlper/gte-large") -> None:
        self.model = SentenceTransformer(model)

    async def compute_embedding(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.model.encode(text))


@deployment
class QueryLegacySearch:
    def __init__(
        self,
        database_name: str = "myntra",
        collection_name: str = "myntra-items-offline",
    ) -> None:
        self.client = AsyncIOMotorClient(os.environ["DB_CONNECTION_STRING"])
        self.database_name = database_name
        self.collection_name = collection_name

    async def run(
        self,
        text_search: Optional[str],
        min_price: int,
        max_price: int,
        min_rating: float,
        n: int = 20,
        text_search_index_name: str = "lexical_text_search_index",
    ) -> list[tuple[str, str]]:
        logger = logging.getLogger("ray.serve")
        logger.setLevel(logging.DEBUG)

        db = self.client[self.database_name]
        collection = db[self.collection_name]

        pipeline = []
        if text_search.strip():
            pipeline.extend(
                lexical_search(
                    text_search=text_search,
                    text_search_index_name=text_search_index_name,
                )
            )

        pipeline.extend(
            match_on_metadata(
                min_price=min_price,
                max_price=max_price,
                min_rating=min_rating,
                n=n,
            )
        )

        logger.debug(f"Running pipeline: {pipeline}")

        records = collection.aggregate(pipeline)
        results = [
            (record["img"].split(";")[-1].strip(), record["name"])
            async for record in records
        ]

        n_results = len(results)
        logger.debug(f"Found {n_results=} results")

        return results


@deployment
class QueryAIEnabledSearch:
    def __init__(
        self,
        embedding_model: DeploymentHandle,
        database_name: str = "myntra",
        collection_name: str = "myntra-items-offline",
    ) -> None:
        self.client = AsyncIOMotorClient(os.environ["DB_CONNECTION_STRING"])
        self.embedding_model = embedding_model
        self.database_name = database_name
        self.collection_name = collection_name

    async def run(
        self,
        text_search: str,
        min_price: int,
        max_price: int,
        min_rating: float,
        categories: list[str],
        colors: list[str],
        seasons: list[str],
        n: int,
        search_type: set[str],
        vector_search_index_name: str = "vector_search_index",
        vector_search_path: str = "description_embedding",
        text_search_index_name: str = "lexical_text_search_index",
        vector_penalty: int = 1,
        full_text_penalty: int = 10,
    ):
        logger = logging.getLogger("ray.serve")
        logger.setLevel(logging.DEBUG)

        db = self.client[self.database_name]
        collection = db[self.collection_name]

        pipeline = []
        if text_search.strip():

            if "vector" in search_type:
                logger.debug(f"Computing embedding for {text_search=}")
                embedding = await self.embedding_model.compute_embedding.remote(
                    text_search
                )

            is_hybrid = search_type == {"vector", "lexical"}
            if is_hybrid:
                pipeline.extend(
                    hybrid_search(
                        self.collection_name,
                        text_search,
                        text_search_index_name,
                        vector_search_index_name,
                        vector_search_path,
                        embedding,
                        n,
                        min_price,
                        max_price,
                        min_rating,
                        categories,
                        colors,
                        seasons,
                        vector_penalty,
                        full_text_penalty,
                    )
                )
            elif search_type == {"vector"}:
                pipeline.extend(
                    vector_search(
                        vector_search_index_name,
                        vector_search_path,
                        embedding,
                        n,
                        min_price,
                        max_price,
                        min_rating,
                        categories,
                        colors,
                        seasons,
                    )
                )
            elif search_type == {"lexical"}:
                pipeline.extend(
                    lexical_search(
                        text_search=text_search,
                        text_search_index_name=text_search_index_name,
                    )
                )
                pipeline.extend(
                    match_on_metadata(
                        min_price=min_price,
                        max_price=max_price,
                        min_rating=min_rating,
                        n=n,
                        categories=categories,
                        colors=colors,
                        seasons=seasons,
                    )
                )
        else:
            pipeline = match_on_metadata(
                min_price=min_price,
                max_price=max_price,
                min_rating=min_rating,
                n=n,
                categories=categories,
                colors=colors,
                seasons=seasons,
            )

        records = collection.aggregate(pipeline)
        logger.debug(f"Running pipeline: {pipeline}")
        records = [record async for record in records]
        results = [
            (record["img"].split(";")[-1].strip(), record["name"]) for record in records
        ]
        num_results = len(results)

        logger.debug(f"Found {num_results=} results")
        return results


fastapi = FastAPI()


@deployment
@ingress(fastapi)
class QueryApplication:

    def __init__(
        self,
        query_legacy: QueryLegacySearch,
        query_ai_enabled: QueryAIEnabledSearch,
    ):
        self.query_legacy = query_legacy
        self.query_ai_enabled = query_ai_enabled

    @fastapi.get("/legacy")
    async def query_legacy_search(
        self,
        text_search: str,
        min_price: int,
        max_price: int,
        min_rating: float,
        num_results: int,
    ):
        return await self.query_legacy.run.remote(
            text_search=text_search,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            n=num_results,
        )

    @fastapi.get("/ai_enabled")
    async def query_ai_enabled_search(
        self,
        text_search: str,
        min_price: int,
        max_price: int,
        min_rating: float,
        categories: list[str],
        colors: list[str],
        seasons: list[str],
        num_results: int,
        embedding_column: str,
        search_type: list[str],
    ):
        logger = logging.getLogger("ray.serve")
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Running query_ai_enabled_search with {locals()=}")
        return await self.query_ai_enabled.run.remote(
            text_search=text_search,
            min_price=min_price,
            max_price=max_price,
            min_rating=min_rating,
            categories=categories,
            colors=colors,
            seasons=seasons,
            n=num_results,
            vector_search_path=f"{embedding_column.lower()}_embedding",
            search_type={type_.lower() for type_ in search_type},
        )


query_legacy = QueryLegacySearch.bind()
embedding_model = EmbeddingModel.bind()
query_ai_enabled = QueryAIEnabledSearch.bind(embedding_model)
app = QueryApplication.bind(query_legacy, query_ai_enabled)
