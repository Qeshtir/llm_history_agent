from abc import ABC, abstractmethod
from typing import List
from langchain_gigachat import GigaChatEmbeddings
from chromadb.api.types import EmbeddingFunction
from config import settings
import logging

logger = logging.getLogger(__name__)


class GigaChatEmbeddingsService(EmbeddingFunction):
    def __init__(self):
        try:
            self.embeddings = GigaChatEmbeddings(
                credentials=settings.GIGACHAT_API_KEY, verify_ssl_certs=False
            )
        except Exception as e:
            logger.error(f"Failed to initialize GigaChat service: {str(e)}")
            raise

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Метод для получения эмбеддингов через GigaChat.
        Реализует интерфейс EmbeddingFunction из ChromaDB.
        """
        try:
            embeddings = [self.embeddings.embed_query(text) for text in input]
            return embeddings
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise


def get_embeddings_service(service_name: str = "gigachat") -> EmbeddingFunction:
    """Фабричный метод для получения сервиса эмбеддингов"""
    services = {
        "gigachat": GigaChatEmbeddingsService,
    }

    if service_name not in services:
        raise ValueError(f"Unknown embedding service: {service_name}")

    return services[service_name]()
