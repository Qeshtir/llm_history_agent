from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from config import settings
from .embeddings import get_embeddings_service

class ChromaService:
    def __init__(self, embedding_service: str = "gigachat"):
        # Инициализация клиента ChromaDB
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        
        # Инициализация сервиса эмбеддингов
        self.embeddings = get_embeddings_service(embedding_service)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Внутренний метод для получения эмбеддингов текстов.
        Инкапсулирует логику работы с различными embedding-моделями и может быть расширен
        для добавления кэширования или батчинга.

        PS (прочтете - удалим):
        Подчеркивание _ в начале имени метода _get_embeddings - это соглашение в Python, 
        которое указывает на то, что метод предназначен 
        для внутреннего использования в классе (условно "приватный" метод).
        """
        return self.embeddings.get_embeddings(texts)

    def create_or_get_collection(self, collection_name: str):
        """Создает новую коллекцию или возвращает существующую"""
        try:
            collection = self.client.get_collection(name=collection_name)
        except ValueError:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Historical documents collection"}
            )
        return collection

    def add_documents(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict] = None,
        ids: List[str] = None
    ):
        """
        Добавляет документы в коллекцию
        
        Args:
            collection_name: Название коллекции
            texts: Список текстов для добавления
            metadatas: Список метаданных для каждого документа
            ids: Список уникальных идентификаторов для документов
        """
        collection = self.create_or_get_collection(collection_name)
        
        if not ids:
            ids = [f"doc_{i}" for i in range(len(texts))]
        
        if not metadatas:
            metadatas = [{"source": "transcript"} for _ in texts]
            
        # Получаем эмбеддинги
        embeddings = self._get_embeddings(texts)
            
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    def query_documents(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 3,
        metadata_filter: Optional[Dict] = None
    ) -> Dict:
        """
        Поиск похожих документов
        
        Args:
            collection_name: Название коллекции
            query_text: Текст запроса
            n_results: Количество результатов
            metadata_filter: Фильтр по метаданным
        
        Returns:
            Dict с результатами поиска
        """
        collection = self.create_or_get_collection(collection_name)
        
        where = metadata_filter if metadata_filter else None
        
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where
        )
        
        return results

    def get_collection_stats(self, collection_name: str) -> Dict:
        """Получение статистики коллекции"""
        collection = self.create_or_get_collection(collection_name)
        return {
            "count": collection.count(),
            "name": collection.name,
            "metadata": collection.metadata
        }

    def delete_collection(self, collection_name: str):
        """Удаление коллекции"""
        self.client.delete_collection(collection_name)

    def update_document_metadata(
        self,
        collection_name: str,
        document_id: str,
        metadata: Dict
    ):
        """Обновление метаданных документа"""
        collection = self.create_or_get_collection(collection_name)
        collection.update(
            ids=[document_id],
            metadatas=[metadata]
        )

    def get_unique_topics(self, collection_name: str) -> List[str]:
        """Получение списка уникальных топиков в коллекции"""
        collection = self.create_or_get_collection(collection_name)
        result = collection.get()
        
        if not result or not result['metadatas']:
            return []
        
        return list(set(meta['topic'] for meta in result['metadatas'] if 'topic' in meta))

    def get_documents_by_topic(self, collection_name: str, topic: str) -> Dict:
        """Получение всех документов по конкретному топику"""
        collection = self.create_or_get_collection(collection_name)
        return collection.query(
            query_texts=[""],
            where={"topic": topic},
            n_results=100  # можно параметризовать если нужно
        )

    def document_exists(self, collection_name: str, file_path: str) -> bool:
        """Проверка существования документа в коллекции"""
        collection = self.create_or_get_collection(collection_name)
        result = collection.query(
            query_texts=[""],
            where={"source": str(file_path)},
            n_results=1
        )
        return len(result['ids']) > 0
