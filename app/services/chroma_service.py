from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from config import settings
from .embeddings import get_embeddings_service
import logging
from uuid import uuid4

logger = logging.getLogger(__name__)


class ChromaService:
    def __init__(self, embedding_service: str = "gigachat"):
        # Инициализация клиента ChromaDB
        self.client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(allow_reset=True, anonymized_telemetry=False),
        )

        # Инициализация сервиса эмбеддингов
        self.embeddings = get_embeddings_service(embedding_service)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Внутренний метод для получения эмбеддингов текстов.
        """
        return self.embeddings(texts)  # Используем __call__ через прямой вызов объекта

    def create_or_get_collection(self, collection_name: str):
        """Создает новую коллекцию или возвращает существующую"""

        logger.info(f"Попытка получить коллекцию: {collection_name}")
        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embeddings,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"Коллекция {collection_name} найдена или создана. Кол-во документов: {collection.count()}"
        )
        return collection

    def add_documents(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict] = None,
        # ids: List[str] = None
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

        uuids = [str(uuid4()) for _ in range(len(texts))]
        # Получаем эмбеддинги
        embeddings = self._get_embeddings(texts)

        # upsert - создает новые документы или обновляет существующие
        collection.upsert(
            documents=texts, embeddings=embeddings, metadatas=metadatas, ids=uuids
        )

        logger.info(f"Добавлено {len(texts)} документов в коллекцию {collection_name}")
        logger.debug(
            f"Добавленные документы: {texts}, метаданные: {metadatas}, идентификаторы: {uuids}"
        )
        logger.info(
            f"Статистика коллекции после добавления: {self.get_collection_stats(collection_name)}"
        )

    def query_documents(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 3,
        metadata_filter: Optional[Dict] = None,
        search_embeding=True,
    ) -> Dict:
        """
        Поиск похожих документов

        Args:
            collection_name: Название коллекции
            query_text: Текст запроса
            n_results: Количество результатов
            metadata_filter: Фильтр по метаданным
            search_embeding:

        Returns:
            Dict с результатами поиска
        """
        collection = self.create_or_get_collection(collection_name)

        where = metadata_filter if metadata_filter else None

        #  TODO if any time left - check where_document - A WhereDocument type dict used to filter by the documents. E.g. https://docs.trychroma.com/reference/py-collection#query

        if search_embeding:
            # Получаем эмбеддинг запроса
            query_embedding = self._get_embeddings([query_text])[0]

            results = collection.query(
                query_embeddings=[query_embedding], n_results=n_results, where=where
            )
        else:
            results = collection.query(
                query_texts=[query_text], n_results=n_results, where=where
            )

        return results

    def get_collection_stats(self, collection_name: str) -> Dict:
        """Получение статистики коллекции"""
        collection = self.create_or_get_collection(collection_name)
        return {
            "count": collection.count(),
            "name": collection.name,
            "metadata": collection.metadata,
        }

    def delete_collection(self, collection_name: str):
        """Удаление коллекции"""
        self.client.delete_collection(collection_name)

    def update_document_metadata(
        self, collection_name: str, document_id: str, metadata: Dict
    ):
        """Обновление метаданных документа"""
        collection = self.create_or_get_collection(collection_name)
        collection.update(ids=[document_id], metadatas=[metadata])

    def get_unique_topics(self, collection_name: str) -> List[str]:
        """Получение списка уникальных топиков в коллекции"""
        collection = self.create_or_get_collection(collection_name)
        result = collection.get()

        if not result or not result["metadatas"]:
            return []

        return list(
            set(meta["topic"] for meta in result["metadatas"] if "topic" in meta)
        )

    def get_documents_by_topic(self, collection_name: str, topic: str) -> Dict:
        """Получение всех документов по конкретному топику"""
        collection = self.create_or_get_collection(collection_name)
        return collection.query(
            query_texts=[""],
            where={"topic": topic},
            n_results=100,  # можно параметризовать если нужно
        )

    def get_documents_by_metadata(
        self, collection_name: str, metadata: Dict, n_results: int = 100
    ) -> Dict:
        """Получение всех документов по конкретным метаданным"""
        collection = self.create_or_get_collection(collection_name)
        return collection.query(query_texts=[""], where=metadata, n_results=n_results)

    def document_exists(self, collection_name: str, file_path: str) -> bool:
        """Проверка существования документа в коллекции"""
        collection = self.create_or_get_collection(collection_name)
        result = collection.query(
            query_texts=[""], where={"path": str(file_path)}, n_results=1
        )
        logger.debug(f"Результаты запроса для {file_path}: {result}")
        return len(result["documents"]) > 0 and len(result["documents"][0]) > 0
