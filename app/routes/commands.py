from aiogram import Router
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.filters.command import Command

from utils.states import ProcessLLMStates
import logging

router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message, state: FSMContext):
    msg_1 = """
    Привет!
    Я своего рода эксперт в русско-японской войне.
    Можем обсудить с тобой эту тему.
    """
    msg_2 = "Задавай свои вопросы, а я расскажу тебе все, что знаю"

    logging.info(f"State set to: {await state.get_state()}")
    await message.answer(msg_1)
    await message.answer(msg_2)
    await state.set_state(ProcessLLMStates.waitForText)
    logging.info(f"State set to: {await state.get_state()}")
