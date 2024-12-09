import os
from pydantic_settings import BaseSettings


class DB_Settings:
    pass


class TG_Settings:
    TG_BOT_TOKEN: str = os.getenv("TG_BOT_TOKEN")


class LLM_Settings:
   pass


class Settings(BaseSettings):
    TG_BOT_TOKEN: str

    GIGACHAT_API_KEY: str
    EMBEDDING_SERVICE: str = "gigachat"
    LLM_GIGACHAT_MODEL: str = "GigaChat-Pro"

    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    # Настройки для обработки текста
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    class Config:
        env_file = ".env"

# Создаем экземпляр настроек
settings = Settings()