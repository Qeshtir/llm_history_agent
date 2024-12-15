import pytest
from services.chroma_service import ChromaService


@pytest.fixture
def chroma_service():
    service = ChromaService()
    # Очищаем тестовую коллекцию перед каждым тестом
    try:
        service.delete_collection("test_collection")
    except:
        pass
    return service


def test_get_embeddings(chroma_service):
    """Тест размерности эмбеддингов"""
    texts = ["Тестовый текст"]
    embeddings = chroma_service._get_embeddings(texts)
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1024  # Обновлено с 1024 на 384


def test_create_collection(chroma_service):
    """Тест создания коллекции"""
    collection = chroma_service.create_or_get_collection("test_collection")
    assert collection is not None
    assert collection.name == "test_collection"


def test_add_and_query_documents(chroma_service):
    """Тест добавления и поиска документов"""
    texts = [
        "Русско-японская война началась в 1904 году.",
        "Цусимское сражение произошло в 1905 году.",
        "Война закончилась подписанием Портсмутского мира.",
    ]

    # Добавляем документы
    chroma_service.add_documents(
        collection_name="test_collection",
        texts=texts,
        metadatas=[{"topic": "russo_japanese_war"} for _ in texts],
    )

    # Ищем похожие документы
    query_text = "Когда произошло Цусимское сражение?"

    results = chroma_service.query_documents(
        collection_name="test_collection", query_text=query_text, n_results=1
    )

    assert len(results["documents"]) == 1
    assert "1905" in results["documents"][0][0]


def test_metadata_filter(chroma_service):
    """Тест фильтрации по метаданным"""
    texts = ["Текст 1", "Текст 2"]
    metadatas = [{"topic": "topic1"}, {"topic": "topic2"}]

    chroma_service.add_documents(
        collection_name="test_collection", texts=texts, metadatas=metadatas
    )

    results = chroma_service.query_documents(
        collection_name="test_collection",
        query_text="Текст",
        metadata_filter={"topic": "topic1"},
    )

    assert len(results["documents"]) == 1
    assert results["documents"][0][0] == "Текст 1"


# def test_semantic_similarity(chroma_service):
#     """Тест семантической близости документов"""
#     texts = [
#         "Русско-японская война началась в 1904 году",
#         "В 1904 году началась война между Россией и Японией",
#         "Олимпийские игры прошли в Париже"  # Семантически далекий текст
#     ]

#     chroma_service = ChromaService(embedding_service="test")

#     chroma_service.add_documents(
#         collection_name="test_collection",
#         texts=texts
#     )

#     # Ищем похожие документы
#     results = chroma_service.query_documents(
#         collection_name="test_collection",
#         query_text="Когда началась война с Японией?",
#         n_results=2
#     )

#     # Проверяем, что первые два текста (о войне) найдены как наиболее релевантные
#     assert all(i in results['documents'][0] for i in ["война", "1904"])
