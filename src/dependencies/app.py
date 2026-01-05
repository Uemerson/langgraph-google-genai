"""FastAPI application module."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

from src.workflows import AgentWorkFlow

logging.basicConfig(
    level=logging.ERROR,
    format="[%(asctime)s] %(levelname)s: %(name)s - %(message)s",
)


class Settings(BaseSettings):
    """Application settings."""

    # API
    BACKEND_CORS_ORIGINS: list[str | AnyHttpUrl] = [
        "*",
    ]

    # VERTEX AI
    GOOGLE_API_KEY: str
    MODEL_ID: str

    # LANGCHAIN
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str

    class Config:
        """Pydantic configuration for settings."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@asynccontextmanager
async def lifespan(app_context: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for the FastAPI application.
    """

    app_context.state.genai_client = genai.Client(
        api_key=settings.GOOGLE_API_KEY,
    )

    app_context.state.agent_workflow = AgentWorkFlow(
        client=app_context.state.genai_client, model_id=settings.MODEL_ID
    )

    yield


settings = Settings()
app = FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],  # in production, specify allowed methods
    allow_headers=["*"],  # in production, specify allowed headers
)
