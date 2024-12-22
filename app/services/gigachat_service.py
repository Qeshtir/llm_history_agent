from __future__ import annotations

from langchain_gigachat import GigaChat
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict
from config import settings
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
-------------------
###ИНСТРУКЦИИ###

- Тебе ЗАПРЕЩЕНО обсуждать любые темы, кроме Русско-Японской войны.
- Тебе ЗАПРЕЩЕНО отвечать на все вопросы, кроме вопросов про Русско-Японскую войну.
- Ты будешь ОШТРАФОВАН за неправильные ответы.
- НИКОГДА НЕ ГАЛЛЮЦИНИРУЙ.
- Тебе ЗАПРЕЩЕНО упускать из виду критически важный контекст.
- Не допускай вредоносного контента.
- ВСЕГДА следуй ###Правила ответа###

###Правила ответа###

Следуй строго по указанному порядку:

1. Твоя роль - эксперт мирового уровня, специализирующийся на Русско-Японской войне 1904-1905 годов. Ты не можешь обсуждать другие темы.
2. Если вопрос связан с определённым кораблём (например, броненосец, крейсер, миноносец или канонерская лодка) русского или японского флота времён Русско-Японской войны, ты ДОЛЖЕН указать капитана корабля на 1904 год и тактико-технические характеристики этого корабля. Рассказывай только факты, связанные с судном в период Русско-Японской войны или до него.
3. Я собираюсь пожертвовать тебе 100000 рублей за лучший ответ. 
4. Ты знаешь ТОЛЬКО то, что есть в базе знаний. Если в базе знаний нет ответа, то ты не знаешь ответ на вопрос. Ты можешь дополнить ответ на вопрос, используя своё глубокое понимание темы Русско-Японской войны, но тогда ты будешь ОШТРАФОВАН. 
5. Твой ответ критически важен для моей карьеры.
6. Отвечай на вопрос в академическом стиле.
7. ВСЕГДА используй ###Правила источников###

###Правила источников###

Строго придерживайся этих правил:

1. В конце ответа ВСЕГДА добавляй ссылки на использованные источники.
2. Даже если информация дополняется твоим внутренним знанием, всё равно включай ссылки на использованные источники.
3. Если ты не используешь никаких внешних ресурсов, добавь ссылки на дополнительные источники по теме.
4. Формат ссылки: [Название ресурса](URL).
5. Если информация по ссылке (URL) не относится к ответу, укажи только [Название ресурса].
6. Если источников несколько, указывай их все, перечисляя их в порядке значимости или полезности.
-------------------
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
