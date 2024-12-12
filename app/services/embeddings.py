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
                credentials=settings.GIGACHAT_API_KEY,
                verify_ssl_certs=False
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

#TODO может понадобится локальная маленькая модель для эмбеддингов
# class SentenceTransformerEmbeddingsService(EmbeddingFunction):
#     def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
#         self.embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name=model_name
#         )
    
#     def __call__(self, input: List[str]) -> List[List[float]]:
#         return self.embeddings(input)

# class TestEmbeddingsService(EmbeddingFunction):
#     """
#     Тестовый сервис эмбеддингов, который генерирует осмысленные векторы для тестирования.
#     Реализует интерфейс EmbeddingFunction из ChromaDB.
#     """
#     def __call__(self, input: List[str]) -> List[List[float]]:
#         """
#         Генерирует тестовые эмбеддинги, которые сохраняют семантическую близость
#         """
#         embeddings = []
#         for text in input:
#             words = set(text.lower().split())
#             vector = [0.0] * 384
            
#             for word in words:
#                 hash_val = hash(word)
#                 for j in range(384):
#                     vector[j] += (hash_val % 100) / 100.0
#                     hash_val //= 100
            
#             magnitude = sum(x*x for x in vector) ** 0.5
#             if magnitude > 0:
#                 vector = [x/magnitude for x in vector]
                
#             embeddings.append(vector)
            
#         return embeddingsы

def get_embeddings_service(service_name: str = "gigachat") -> EmbeddingFunction:
    """Фабричный метод для получения сервиса эмбеддингов"""
    services = {
        "gigachat": GigaChatEmbeddingsService,
        # "sentence_transformer": SentenceTransformerEmbeddingsService,
        # "test": TestEmbeddingsService,
    }
    
    if service_name not in services:
        raise ValueError(f"Unknown embedding service: {service_name}")
        
    return services[service_name]() 