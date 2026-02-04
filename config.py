# config.py
from pydantic_settings import BaseSettings
from cryptography.fernet import Fernet
import os

class Settings(BaseSettings):
    # API Keys (required in .env)
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str = ""

    # Database settings
    DATABASE_URL: str = "sqlite:///./persona_clone.db"
    VECTOR_DB_PATH: str = "./vector_db"

    # Model settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL: str = "gpt-3.5-turbo"
    REWARD_MODEL: str = "microsoft/DialoGPT-medium"

    # Reinforcement learning settings
    RL_LEARNING_RATE: float = 1e-4
    RL_BATCH_SIZE: int = 32
    RL_GAMMA: float = 0.99
    MAX_TRAINING_STEPS: int = 10000

    # Privacy & security
    ENCRYPTION_KEY: str = None  # Will auto-generate if missing
    DATA_RETENTION_DAYS: int = 730  # 2 years

    # Interview settings
    MAX_QUESTIONS_PER_CATEGORY: int = 10
    MIN_RESPONSE_LENGTH: int = 20
    INTERVIEW_TIMEOUT_MINUTES: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Generate encryption key if not provided
        if not self.ENCRYPTION_KEY or self.ENCRYPTION_KEY in ["", "generate_secure_key_here"]:
            self.ENCRYPTION_KEY = Fernet.generate_key().decode()

# Global settings instance
settings = Settings()
