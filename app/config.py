import os
from pydantic import BaseSettings


class DB_Settings:
    pass


class TG_Settings:
    TG_BOT_TOKEN: str = os.getenv("TG_BOT_TOKEN")


class LLM_Settings:
   pass


class Settings(BaseSettings):
    TG_BOT_TOKEN: str
    GIGACHAT_API_KEY: str
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_db"
    EMBEDDING_SERVICE: str = "gigachat"

    class Config:
        env_file = ".env"