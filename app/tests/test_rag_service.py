import pytest
import logging
import json
from services.rag_service import RAGService
from services.chroma_service import ChromaService
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)  # Установите уровень логирования на INFO или DEBUG
logger = logging.getLogger(__name__)

@pytest.fixture
def rag_service():
    service = RAGService()
    # Очищаем тестовую коллекцию перед каждым тестом
    try:
        service.delete_collection("test_collection")
    except:
        pass
    return service

def create_test_file(docs_dir: str, filename: str, content: str):
    """Создает тестовый файл с заданным содержимым."""
    file_path = Path(docs_dir) / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Создан тестовый файл: {file_path}")

def create_test_urls_json(docs_dir: str, filename: str, urls_data: list):
    """Создает тестовый файл urls.json с тестовыми данными."""
    file_path = Path(docs_dir) / filename
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(urls_data, f, ensure_ascii=False, indent=4)
    logger.info(f"Создан тестовый файл: {file_path}")

def test_load_documents_from_directory(rag_service):
    """Тест загрузки документов из директории в ChromaDB"""
    docs_dir = "/app/data/scratches/test"
    collection_name = "test_collection"
    
    # Убедимся, что директория существует
    Path(docs_dir).mkdir(parents=True, exist_ok=True)

    # Создаем тестовый файл, если его нет
    test_filename = "test_document.txt"
    test_content = "Это тестовый документ для проверки загрузки в ChromaDB."
    create_test_file(docs_dir, test_filename, test_content)

    # Создаем тестовый файл urls.json, если его нет
    urls_json_filename = "test_urls.json"
    urls_data = [
        {
            "file": test_filename,
            "url": "https://www.test.com/test"
        }
    ]
    create_test_urls_json(docs_dir, urls_json_filename, urls_data)

    # Загружаем документы
    rag_service.load_documents_from_directory(docs_dir, urls_json_filename, collection_name)

    # Получаем статистику коллекции
    stats = rag_service.chroma_service.get_collection_stats(collection_name)
    logger.info(f"Статистика коллекции: {stats}")
    assert stats['count'] > 0  # Проверяем, что документы были загружены
    
    # Проверяем поиск и метаданные
    results = rag_service.chroma_service.query_documents(
        collection_name=collection_name,
        query_text="тестовый",
        n_results=20
    )

    logger.info(f"Результаты поиска: {results}")
    logger.info(f"Размер результатов: {len(results['documents'][0])}")
    
    assert len(results['documents'][0]) > 0
    assert len(results['metadatas'][0]) > 0
    assert 'path' in results['metadatas'][0][0]

def test_generate_answer(rag_service):
    """Тест генерации ответа на основе запроса"""

    documents = [
        "Документ 1: Информация о Русско-японской войне.",
        "Документ 2: Цусимское сражение произошло в 1905 году."
    ]
    collection_name = "test_collection"
    
    # Добавляем документы в коллекцию
    rag_service.chroma_service.add_documents(
        collection_name=collection_name,
        texts=documents,
        metadatas=[{"topic": "test_topic"} for _ in documents]
    )

    query = "Когда произошло Цусимское сражение?"
    answer = rag_service.generate_answer(query, collection_name)
    
    # Проверяем, что ответ содержит ключевую информацию
    assert "1905" in answer 