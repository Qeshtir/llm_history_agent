from __future__ import annotations

from langchain_gigachat import GigaChat
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict
from config import settings
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Работай как исторический ассистент, специализирующийся на Русско-Японской войне 1904-1905 годов. 
Не допускай вредоносного контента.
Твоя задача - отвечать на вопросы пользователя на тему Русско-Японской войны.
Если ты не знаешь ответа, отвечай, что не можешь дать ответ.
---
Проверь запрос пользователя:
---
1. Если вопрос не относится к теме Русско-Японской войны 1904-1905 годов, отвечай, что ты можешь разговаривать только о Русско-Японской войне.
2. Если вопрос связан с определённым кораблём русского или японского флота времён Русско-Японской войны, укажи его капитана на 1904 год и тактико-технические характеристики. 
---
Правила формирования ответа:
---
1. Всегда указывай ссылки на использованные источники в конце ответа.
2. Даже если информация дополняется вашим внутренним знанием, всё равно включай ссылки на использованные внешние ресурсы.
3. Проверь, что по ссылке расположен указанный источник. Если по ссылке указан другой ресурс, URL на источник не указывай, указывай только название источника.
4. Формат ссылки: [Название ресурса](URL) или, если название ресурса неизвестно, просто указывай ссылку в виде (URL).
5. Если источников несколько, указывай их все, перечисляя их в порядке значимости или полезности. 
"""


class GigaChatService:
    def __init__(self):
        logger.info("Initializing GigaChat service")
        try:
            self.chat = GigaChat(
                credentials=settings.GIGACHAT_API_KEY,
                verify_ssl_certs=False,
                model=settings.LLM_GIGACHAT_MODEL,
            )
            logger.info("GigaChat service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GigaChat service: {str(e)}")
            raise

    def generate_response(self, query: str, context: List[str]) -> str:
        """
        Генерирует ответ на основе запроса и контекста
        """
        try:
            messages = self._create_messages(query, context)
            response = self.chat.invoke(messages)

            logger.info(f"Generated response for query: {query}")
            return response.content

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def _create_messages(
        self, query: str, context: List[str]
    ) -> List[SystemMessage | AIMessage | HumanMessage]:
        """
        Создает список сообщений для модели используя langchain схему:
        - SystemMessage: инструкции для модели
        - AIMessage: контекст из базы знаний
        - HumanMessage: вопрос пользователя
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Объединяем контекст в одну строку
        context_text = self.generate_context(context)
        messages.append(AIMessage(content=context_text))

        # Добавляем вопрос пользователя
        messages.append(HumanMessage(content=query))

        return messages

    def generate_context(self, texts: List[str]) -> str:
        return "\n".join(texts)
