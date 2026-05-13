from typing import List

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    database_url: str
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    soft_limit_chars: int = 1800
    frontend_origins: List[AnyHttpUrl] = ["http://localhost:5173"]
    mlflow_model_path: str = "/app/models/mlflow/artifacts/models/m-b48b9f64e6b64ef888ae1c54458ac274"


settings = Settings()
