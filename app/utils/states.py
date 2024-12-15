from aiogram.fsm.state import StatesGroup, State


class ProcessLLMStates(StatesGroup):
    waitForText = State()
