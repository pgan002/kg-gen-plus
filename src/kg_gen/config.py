from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    token_url: Optional[str] = None
    llm_model: str = "openai/gpt-5-nano"
    llm_api_key: Optional[str] = None
    llm_temperature: float = 1.0
    retrieval_model: str = "all-MiniLM-L6-v2"
    api_base: Optional[str] = None

    # model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="allow")


settings = Settings()
