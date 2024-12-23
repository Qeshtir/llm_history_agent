import pytest
from pathlib import Path
from services.text_processor import TextProcessor


@pytest.fixture
def processor():
    return TextProcessor(chunk_size=100, chunk_overlap=20)


def test_validate_chunk_params():
    """Тест валидации параметров чанков"""
    # Правильные параметры
    TextProcessor(chunk_size=100, chunk_overlap=20)

    # Неправильные параметры
    with pytest.raises(ValueError):
        TextProcessor(chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        TextProcessor(chunk_size=50, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextProcessor(chunk_size=100, chunk_overlap=-10)


def test_clean_text(processor):
    """Тест очистки текста"""
    text = "Привет,   мир!\n\nКак  дела?\t\tХорошо!"
    cleaned = processor.clean_text(text)
    assert cleaned == "Привет, мир! Как дела? Хорошо!"


def test_split_into_chunks(processor):
    """Тест разбиения на чанки"""
    text = "Это первое предложение. Это второе предложение. А это уже третье."
    chunks = processor.split_into_chunks(text)

    # Проверяем количество чанков
    assert len(chunks) > 0

    # Проверяем размер чанков
    for chunk in chunks:
        assert len(chunk) <= processor.chunk_size

    # Проверяем перекрытие
    if len(chunks) > 1:
        overlap = len(set(chunks[0]) & set(chunks[1]))
        assert overlap > 0


def test_extract_metadata(processor):
    """Тест извлечения метаданных"""
    file_path = Path("test_file.txt")
    metadata = processor.extract_metadata("Тестовый текст", file_path)

    assert "path" in metadata
    assert "length" in metadata
    assert "chunks" in metadata

    # Тест с cleaned_ префиксом
    file_path = Path("cleaned_varyag_1.txt")
    metadata = processor.extract_metadata("Тестовый текст", file_path)
    assert metadata["topic"] == "varyag"
    assert metadata["filename"] == "varyag_1"
