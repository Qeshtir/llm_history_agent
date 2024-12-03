import asyncio
from aiogram import Bot, Dispatcher
from config import TG_Settings


from routes import ml, commands

token = TG_Settings.TG_BOT_TOKEN

bot = Bot(token=token)
dp = Dispatcher()


async def start():
    try:
        await dp.start_polling(bot, skip_updates=True)
    finally:
        await bot.session.close()


dp.include_router(commands.router)
dp.include_router(ml.router)


if __name__ == "__main__":
    asyncio.run(start())