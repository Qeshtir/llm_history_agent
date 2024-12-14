from aiogram import Router
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

from utils.states import ProcessLLMStates
from services.rag_service import RAGService
import os
from dotenv import load_dotenv

rag_service = RAGService()
load_dotenv()
collection_name = os.getenv("CHROMA_COLLECTION_NAME")

router = Router()


@router.message(ProcessLLMStates.waitForText)
async def request_generate(message: Message, state: FSMContext):
    query = message.text

    answer = rag_service.generate_answer(query, collection_name)

    await message.answer(answer)
