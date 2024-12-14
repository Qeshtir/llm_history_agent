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
Здравия желаю, Ваше Превосходительство!

Я - жалкая тень великого адмирала Степана Осиповича Макарова.

Но я могу ответить на Ваши вопросы по Русско-Японской войне 1904-05гг.
"""

    logging.info(f"State set to: {await state.get_state()}")
    await message.answer(msg_1)
    await state.set_state(ProcessLLMStates.waitForText)
    logging.info(f"State set to: {await state.get_state()}")
