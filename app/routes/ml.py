from aiogram import Router
from aiogram.types import Message
from aiogram.fsm.context import FSMContext

from utils.states import ProcessLLMStates

router = Router()


@router.message(ProcessLLMStates.waitForText)
async def request_generate(message: Message, state: FSMContext):
    user_text = message.text
    
    answer = "вот тебе умный развернутый ответ"

    await message.answer(answer)