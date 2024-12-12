import logging
from pathlib import Path
from services.text_processor import TextProcessor
from services.chroma_service import ChromaService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(docs_dir: str, collection_name: str = "history_lectures"):
    processor = TextProcessor()
    chroma_service = ChromaService()
    
    docs_path = Path(docs_dir)
    for file_path in docs_path.glob("*.txt"):
        logger.info(f"Обработка файла: {file_path}")
        try:
            # # Проверяем, не загружен ли уже документ
            # if chroma_service.document_exists(collection_name, str(file_path)):
            #     logger.info(f"Документ {file_path} уже существует в коллекции")
            #     continue
                
            # Обрабатываем и загружаем документ
            chunks, metadata_list = processor.process_file(file_path)
            logger.info(f"Загружаем {len(chunks)} чанков в коллекцию {collection_name}")
            chroma_service.add_documents(
                collection_name=collection_name,
                texts=chunks,
                metadatas=metadata_list
            )
            logger.info(f"Документ {file_path} успешно загружен")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке {file_path}: {str(e)}")

if __name__ == "__main__":
    load_documents("scratches/cleaned_docs") 