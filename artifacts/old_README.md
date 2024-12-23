# llm_history_agent
Мини Улиточка на страже истории! Выпускной проект курса "Введение в LLM".


## пусть пока это все будет тут

```docker-compose run -e PYTHONPATH=/app bot pytest -v tests/test_rag_service.py -v --log-cli-level=DEBUG```
запуск теста test_rag_service.py с логами в консоле

```docker-compose run -e PYTHONPATH=/app bot pytest -v tests``` 
 без логов


 ```python
""""Пример генерации ответа на основе запроса"""

from services.rag_service import RAGService

rag_service = RAGService()

collection_name = "russo_japanese_war"

query = "Когда произошло Цусимское сражение?"
answer = rag_service.generate_answer(query, collection_name)

# Проверяем, что ответ содержит ключевую информацию
assert "1905" in answer 

```


```python

"""Пример загрузки документов из директории в ChromaDB"""

from services.rag_service import RAGService

rag_service = RAGService()

docs_dir = "/app/data/scratches/cleaned_docs"
collection_name = "russo_japanese_war"
json_file = "urls.json" #FIXME: в конфиг?

# Загружаем документы
rag_service.load_documents_from_directory(docs_dir, json_file, collection_name)

# Получаем статистику коллекции
stats = rag_service.chroma_service.get_collection_stats(collection_name)
logger.info(f"Статистика коллекции: {stats}")

if (stats['count'] > 0):  # Проверяем, что документы были загружены

    # Проверяем поиск и метаданные
    results = rag_service.chroma_service.query_documents(
        collection_name=collection_name,
        query_text="Япония",
        n_results=20
    )

    logger.info(f"Результаты поиска: {results}")
    logger.info(f"Размер результатов: {len(results['documents'][0])}")
    
    logger.info(f"Результаты : количество документов(тип в хроме) { len(results['documents'][0])}, должно совподать с к-вом метаданных  {len(results['metadatas'][0])}, самый первый path = {results['metadatas'][0][0]['path']}")
else:
    logger.error(f"Документы не загружены")

```