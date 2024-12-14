import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class DB_Settings:
    pass


class TG_Settings:
    TG_BOT_TOKEN: str = os.getenv("TG_BOT_TOKEN")


class LLM_Settings:
   pass

#FIXME: хочу видеть в таком виде, в др сестах перейти на него
class Settings(BaseSettings):
    TG_BOT_TOKEN: str = os.getenv("TG_BOT_TOKEN")

    GIGACHAT_API_KEY: str = os.getenv("GIGACHAT_API_KEY")
    EMBEDDING_SERVICE: str = "gigachat"
    LLM_GIGACHAT_MODEL: str = "GigaChat-Pro"

    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    # Настройки для обработки текста
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 50
    
    COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME")

    class Config:
        env_file = ".env"

# Создаем экземпляр настроек
settings = Settings()