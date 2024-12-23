import os
import json
from services.chroma_service import ChromaService
from services.gigachat_service import GigaChatService
from typing import List, Dict
from pathlib import Path
from services.text_processor import TextProcessor
import logging
from dotenv import load_dotenv

load_dotenv()
collection_name = os.getenv("CHROMA_COLLECTION_NAME")

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self):
        self.chroma_service = ChromaService()
        self.giga_chat_service = GigaChatService()

    def load_documents_from_directory(
        self,
        docs_dir: str = "/app/data/scratches/cleaned_docs",
        json_file: str = "urls.json",
        collection_name: str = collection_name,
    ):
        """
        Загружает документы из указанной директории в ChromaDB.

        Args:
            docs_dir: Путь к директории с документами.
            json_file: Путь к JSON-файлу с URL.
            collection_name: Название коллекции в ChromaDB.
        """

        # FIXME: причасать функцию, возможно вывести в сервис

        processor = TextProcessor()
        urls_path = Path(docs_dir) / json_file

        logger.info(f"Загрузка документов из директории: {docs_dir}")
        logger.info(f"Путь к JSON файлу: {urls_path}")

        # Проверяем существование директории и файла
        if not Path(docs_dir).exists():
            logger.error(f"Директория не существует: {docs_dir}")
            raise FileNotFoundError(f"Directory not found: {docs_dir}")

        if not urls_path.exists():
            logger.error(f"JSON файл не найден: {urls_path}")
            raise FileNotFoundError(f"JSON file not found: {urls_path}")

        # Загружаем URL из JSON-файла
        with open(urls_path, "r", encoding="utf-8") as f:
            urls = json.load(f)
            logger.info(f"Загружено {len(urls)} URL из JSON файла")
            logger.debug(f"Содержимое urls: {urls}")  # Добавляем вывод содержимого

        for doc in urls:
            file_path = Path(docs_dir) / doc["file"]
            logger.info(f"Обработка файла: {file_path}")

            # Проверяем существование файла
            if not file_path.is_file():
                logger.warning(f"Файл не найден: {file_path}")
                continue

            # Проверяем, не загружен ли уже документ
            # FIXME: if any time left добавить проверку на дублирование документов, Например, по метаданным

            try:
                # Обрабатываем и загружаем документ
                chunks, metadata_list = processor.process_file(file_path)
                logger.info(f"Получено {len(chunks)} чанков из файла {doc['file']}")
                if not chunks:
                    logger.warning(f"Не получено чанков из файла {doc['file']}")
                    continue

                # Добавляем URL в метаданные каждого чанка
                for metadata in metadata_list:
                    metadata["source"] = doc["url"]

                # Загружаем документы в ChromaDB
                self.chroma_service.add_documents(
                    collection_name=collection_name,
                    texts=chunks,
                    metadatas=metadata_list,
                )
                logger.info(
                    f"Документы успешно загружены в коллекцию {collection_name}"
                )

            except Exception as e:
                logger.error(f"Ошибка при обработке файла {doc['file']}: {str(e)}")
                raise

    def generate_answer(self, query: str, collection_name: str) -> str:
        """
        Генерирует ответ на основе запроса, используя RAG-логику.

        Args:
            query: Вопрос пользователя.
            collection_name: Название коллекции в ChromaDB.

        Returns:
            str: Сгенерированный ответ.
        """
        # Поиск релевантных документов
        results = self.chroma_service.query_documents(
            collection_name=collection_name, query_text=query, n_results=5
        )

        # Извлечение контекста
        context = [
            doc[0] for doc in results["documents"]
        ]  # Извлекаем текст из каждого документа

        # Генерация ответа
        answer = self.giga_chat_service.generate_response(query, context)

        return answer
