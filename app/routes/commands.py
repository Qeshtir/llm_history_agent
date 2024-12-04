from aiogram import Router
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.filters.command import Command

from utils.states import ProcessLLMStates

router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    msg = """
    Привет!
    Я своего рода эксперт в русско-японской войне.
    Можем обсудить с тобой эту тему.
    """
    await message.answer(msg)
    await cmd_bot(message, state)


@router.message()
async def cmd_bot(message: Message, state: FSMContext):
    msg = "Задавай свои вопросы, а я расскажу тебе все, что знаю"
    await message.answer(msg)
    await state.set_state(ProcessLLMStates.waitForText)
