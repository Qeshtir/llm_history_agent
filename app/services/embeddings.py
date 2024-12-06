from abc import ABC, abstractmethod
from typing import List
from langchain_gigachat import GigaChatEmbeddings
from chromadb.utils import embedding_functions
from config import settings

class BaseEmbeddings(ABC):
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

class GigaChatEmbeddingsService(BaseEmbeddings):
    def __init__(self):
        self.embeddings = GigaChatEmbeddings(
            credentials=settings.GIGACHAT_CREDENTIALS,
            verify_ssl_certs=False
        )
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [self.embeddings.embed_query(text) for text in texts]

#TODO может понадобится локальная маленькая модель для эмбеддингов
# class SentenceTransformerEmbeddingsService(BaseEmbeddings):
#     def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
#         self.embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name=model_name
#         )
    
#     def get_embeddings(self, texts: List[str]) -> List[List[float]]:
#         return self.embeddings(texts)

def get_embeddings_service(service_name: str = "gigachat") -> BaseEmbeddings:
    """Фабричный метод для получения сервиса эмбеддингов"""
    services = {
        "gigachat": GigaChatEmbeddingsService,
        # "sentence_transformer": SentenceTransformerEmbeddingsService
    }
    
    if service_name not in services:
        raise ValueError(f"Unknown embedding service: {service_name}")
        
    return services[service_name]() 