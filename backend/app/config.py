from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: SecretStr = SecretStr("")
    ingest_database_url: SecretStr = SecretStr("")
    app_env: Literal["local", "production"] = "local"
    dev_auth_enabled: bool = False
    dev_user_id: int = Field(default=1, gt=0)
    dev_api_token: SecretStr = SecretStr("")
    dev_admin_token: SecretStr = SecretStr("")
    max_concurrent_generations: int = Field(default=4, ge=1, le=64)
    memory_enabled: bool = True
    memory_max_turns: int = Field(default=6, ge=1, le=20)
    memory_max_chars: int = Field(default=6000, ge=1000, le=6000)
    memory_rewrite_max_tokens: int = Field(default=384, ge=64, le=1024)
    memory_summary_enabled: bool = False
    summary_llm_provider: Literal['','ollama','openai','lmstudio'] = ''
    summary_llm_base_url: str = ''
    summary_llm_model: str = ''
    summary_llm_api_key: SecretStr = SecretStr('')
    summary_input_chars: int = Field(default=4000,ge=3000,le=12000)
    summary_max_chars: int = Field(default=1000,ge=200,le=2000)
    summary_max_tokens: int = Field(default=384,ge=128,le=1024)
    embedding_base_url: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = Field(default=1024, ge=1, le=2000)
    llm_provider: Literal["ollama", "openai", "lmstudio"] = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "qwen3.5:9b"
    llm_context_tokens: int = Field(default=8192, ge=4096)
    llm_max_tokens: int = Field(default=1000, ge=64, le=2048)
    http_timeout_seconds: float = Field(default=180, gt=0)
    generation_context_chars: int = Field(default=14000,ge=4000,le=48000)
    deep_llm_model: str = ''
    deep_llm_base_url: str = ''
    deep_llm_provider: Literal['','ollama','openai','lmstudio'] = ''
    deep_llm_api_key: SecretStr = SecretStr('')
    deep_structured_output: bool = True
    deep_max_steps: int = Field(default=12,ge=4,le=30)
    deep_max_searches: int = Field(default=3,ge=1,le=6)
    deep_retry_limit: int = Field(default=1,ge=0,le=3)
    deep_timeout_seconds: float = Field(default=240,ge=5,le=900)
    deep_max_tokens: int = Field(default=1500,ge=256,le=2048)
    deep_context_chars: int = Field(default=14000,ge=4000,le=48000)
    deep_context_tokens: int = Field(default=8192,ge=4096,le=65536)

    @model_validator(mode="after")
    def validate_auth(self):
        if self.memory_summary_enabled and self.summary_input_chars < self.summary_max_chars+2500:
            raise ValueError('SUMMARY_INPUT_CHARS must fit SUMMARY_MAX_CHARS plus at least 2500 characters of new dialogue')
        if self.app_env == "production":
            raise ValueError("Production auth is not implemented; use local mode only")
        if self.dev_auth_enabled:
            token = self.dev_api_token.get_secret_value()
            if len(token) < 24 or token.startswith("CHANGE_ME"):
                raise ValueError("Set a random DEV_API_TOKEN of at least 24 characters")
        return self
