import asyncio
from aiogram import Bot, Dispatcher
from config import TG_Settings
from utils.commands import set_commands

from routes import ml, commands
import logging

logging.basicConfig(level=logging.INFO)
token = TG_Settings.TG_BOT_TOKEN

bot = Bot(token=token)
dp = Dispatcher()

dp.include_router(commands.router)
dp.include_router(ml.router)

async def start():
    try:
        await set_commands(bot)
        await dp.start_polling(bot, skip_updates=True)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(start())