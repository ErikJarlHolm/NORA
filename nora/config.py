"""Configuration for NORA, loaded from environment variables or .env file."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Azure AI Foundry
    project_endpoint: str = ""
    model_deployment_name: str = "gpt-5-mini"
    agent_name: str = "nora"

    # Data folder
    data_folder: Path = Path(
        r"C:\Users\erikholm\OneDrive - Atea\Documents\Kunder\Atea AI Norge\Agent tallknusing"
    )

    # Logging
    log_level: str = "INFO"


settings = Settings()
