import pytest
from pathlib import Path
from scripts.load_documents import load_documents
from services.chroma_service import ChromaService


def test_load_documents(tmp_path):
    """Тест полного пайплайна загрузки документов"""
    # Создаем тестовые файлы
    test_dir = tmp_path / "test_docs"
    test_dir.mkdir()

    test_file = test_dir / "cleaned_test_1.txt"
    test_file.write_text("Это тестовый документ для проверки загрузки.")

    # Загружаем документы
    load_documents(str(test_dir), "test_collection")

    # Проверяем, что файл был создан
    # assert test_file.exists()

    # Проверяем загрузку
    chroma_service = ChromaService()
    results = chroma_service.query_documents(
        collection_name="test_collection", query_text="тестовый документ"
    )

    assert len(results["documents"]) > 0
