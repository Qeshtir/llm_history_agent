from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re
import logging
from config import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, chunk_size: int = settings.CHUNK_SIZE, 
                 chunk_overlap: int = settings.CHUNK_OVERLAP):
        """
        Инициализация процессора текстов
        
        Args:
            chunk_size: Размер чанка текста (в символах)
            chunk_overlap: Размер пересечения между чанками
        """
        self.validate_chunk_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    @staticmethod
    def validate_chunk_params(chunk_size: int, chunk_overlap: int):
        """Валидация параметров чанков"""
        if chunk_size <= 0:
            raise ValueError("Размер чанка должен быть положительным")
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap не может быть больше размера чанка")
        if chunk_overlap < 0:
            raise ValueError("Overlap не может быть отрицательным")

    def get_text_stats(self, text: str) -> Dict:
        """Получение статистики по тексту"""
        chunks = self.split_into_chunks(text)
        return {
            "total_length": len(text),
            "chunks_count": len(chunks),
            "avg_chunk_size": sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0,
            "min_chunk_size": min(len(chunk) for chunk in chunks) if chunks else 0,
            "max_chunk_size": max(len(chunk) for chunk in chunks) if chunks else 0
        }

    def read_file(self, file_path: Path) -> str:
        """Чтение файла с определением кодировки"""
        encodings = ['utf-8', 'windows-1251', 'cp1251', 'latin1']

        if not file_path.is_file():
            logger.error(f"Файл не найден: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    # if content is None:
                    #     logger.error(f"Не удалось прочитать файл {file_path} ни в одной кодировке")
                    #     continue
                    # if not content.strip():
                    #     logger.warning(f"Файл {file_path} пустой!")
                    #     continue
                    return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Ошибка при чтении файла {file_path}: {str(e)}")
                continue
        
        raise ValueError(f"Не удалось прочитать файл {file_path}")

    def clean_text(self, text: str) -> str:
        """
        Очистка текста от лишних символов и форматирования
        """
        # Удаление множественных пробелов и переносов строк
        text = re.sub(r'\s+', ' ', text)
        # Удаление специальных символов, оставляя пунктуацию
        text = re.sub(r'[^\w\s\.,!?;:()-]', '', text)
        return text.strip()

    def split_into_chunks(self, text: str) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,  chunk_overlap=self.chunk_overlap, strip_whitespace=True)
        chunks = text_splitter.split_text(text)
        return chunks

    def extract_metadata(self, text: str, file_path: Optional[Path] = None) -> Dict:
        """
        Извлечение метаданных из текста и пути к файлу.
        На данный момент извлекаем только:
        - Имя файла как path
        - Тему (название файла без номера) как topic
        """
        metadata = {
            "path": str(file_path) if file_path else "unknown",
            "length": len(text)
        }
        
        if not file_path:
            return metadata

        # Извлекаем имя файла без расширения
        filename = file_path.stem.lower()
        
        # Убираем префикс cleaned_ если есть
        if filename.startswith("cleaned_"):
            filename = filename[8:]  # длина "cleaned_"
        
        # Извлекаем тему, убирая номер в конце (_1, _2 итд)
        topic = re.sub(r'_\d+$', '', filename)
        
        metadata.update({
            "topic": topic,
            "filename": filename
        })
        
        # Убедитесь, что возвращаем только простые типы
        return {
            "path": metadata["path"],
            "topic": metadata["topic"],
            "filename": metadata["filename"],
            "length": metadata["length"]
        }

    def process_file(self, file_path: Path) -> Tuple[List[str], List[Dict]]:
        """
        Полная обработка файла: чтение, очистка, разбиение на чанки и извлечение метаданных
        
        Returns:
            Tuple[List[str], List[Dict]]: (chunks, metadata_list)
        """
        logger.info(f"Начало обработки файла: {file_path}")
        
        try:
            text = self.read_file(file_path)
            logger.debug(f"Файл прочитан, размер: {len(text)} символов")
            
            cleaned_text = self.clean_text(text)
            logger.debug(f"Текст очищен, новый размер: {len(cleaned_text)} символов")
            
            chunks = self.split_into_chunks(cleaned_text)
            logger.debug(f"Текст разбит на {len(chunks)} чанков")
            
            stats = self.get_text_stats(cleaned_text)
            logger.info(f"Статистика обработки: {stats}")
            
            base_metadata = self.extract_metadata(cleaned_text, file_path)
            # base_metadata.update({"text_stats": stats})
            
            metadata_list = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    'path': f"{base_metadata['path']}",
                    'topic': f"{base_metadata['topic']}",
                    'filename': f"{base_metadata['filename']}",
                    'chunk_id': f"{i}",
                    'chunk_total': f"{len(chunks)}",
                    'chunk_text_length': f"{len(chunk)}"
                }
                metadata_list.append(chunk_metadata)
            
            # # Создаем строку со статистикой
            # stats_string = f"Total length: {len(cleaned_text)}, Chunks count: {len(chunks)}, Avg chunk size: {len(cleaned_text) / len(chunks) if chunks else 0:.2f}"
            # logger.info(f"Статистика обработки: {stats_string}")
            
            logger.info(f"Обработка файла {file_path} завершена успешно")
            return chunks, metadata_list
            
        except Exception as e:
            logger.error(f"Ошибка при обработке файла {file_path}: {str(e)}")
            raise 