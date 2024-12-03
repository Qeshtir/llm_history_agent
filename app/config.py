import os


class DB_Settings:
    pass


class TG_Settings:
    TG_BOT_TOKEN: str = os.getenv("TG_BOT_TOKEN")


class LLM_Settings:
   pass