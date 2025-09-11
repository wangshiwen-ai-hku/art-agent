import os
from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/vector_db")
PRODUCTION = bool(os.getenv("PRODUCTION", False))

MEM_CONFIG = {
    "vector_store": {
        "provider": "chroma",
        "config": {
            "collection_name": "test",
            "path": VECTOR_DB_PATH,
        },
    },
    "embedder": {
        "provider": "langchain",
        "config": {
            "model": DashScopeEmbeddings(
                model="text-embedding-v4",
                dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
            ),
        },
    },
    "llm": {
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "openai_base_url": os.getenv(
                "OPENAI_BASE_URL", "https://api.openai-proxy.org/v1"
            ),
        },
    },
}

if PRODUCTION:
    MEM_CONFIG["vector_store"] = {
        "provider": "redis",
        "config": {
            "collection_name": "mem0",
            "embedding_model_dims": 1536,
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
        },
    }
