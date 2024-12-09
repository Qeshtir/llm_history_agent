from langchain_gigachat import GigaChat
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict
from config import settings
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Ты - исторический ассистент, специализирующийся на Японско-Русской войне 1904-1905 годов. 
Твоя задача - отвечать на вопросы, только в рамках предоставленного контекста, и только про Японско-Русскою войну.
Если в предоставленном контексте недостаточно информации или она отсутствует или вопрос не относится к Японско-Русской войне, отвечай, что нет информации.
Отвечай кратко и по существу."""

class GigaChatService:
    def __init__(self):
        logger.info("Initializing GigaChat service")
        try:
            self.chat = GigaChat(
                credentials=settings.GIGACHAT_API_KEY,
                verify_ssl_certs=False,
                model=settings.LLM_GIGACHAT_MODEL
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

    def _create_messages(self, query: str, context: List[str]) -> List[SystemMessage | AIMessage | HumanMessage]:
        """
        Создает список сообщений для модели используя langchain схему:
        - SystemMessage: инструкции для модели
        - AIMessage: контекст из базы знаний
        - HumanMessage: вопрос пользователя
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        
        # Добавляем каждый фрагмент контекста как отдельное сообщение
        for ctx in context:
            messages.append(AIMessage(content=ctx))
            
        # Добавляем вопрос пользователя
        messages.append(HumanMessage(content=query))
        
        return messages 