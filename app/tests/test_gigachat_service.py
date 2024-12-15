import pytest
import re
from services.gigachat_service import GigaChatService


@pytest.fixture
def gigachat_service():
    return GigaChatService()


def test_generate_response(gigachat_service):
    """Тест генерации ответа"""
    query = "Когда произошло Цусимское сражение?"
    context = [
        "Цусимское сражение произошло 27-28 мая 1905 года.",
        "В ходе битвы японский флот под командованием адмирала Того разгромил российскую эскадру.",
    ]

    response = gigachat_service.generate_response(query, context)

    # Проверяем, что ответ содержит ключевую информацию
    assert "1905" in response
    assert re.search(
        r"\bмай\b|\bмая\b", response.lower()
    )  # Проверяем наличие "май" или "мая"


def test_generate_response_no_context(gigachat_service):
    """Тест генерации ответа без релевантного контекста"""
    query = "Кто изобрел телефон?"
    context = ["Цусимское сражение произошло в 1905 году."]

    response = gigachat_service.generate_response(query, context)

    # Проверяем, что модель признает отсутствие информации
    assert (
        "нет информации" in response.lower() or "не могу ответить" in response.lower()
    )
