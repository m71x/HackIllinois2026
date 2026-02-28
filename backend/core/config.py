from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    cerebras_api_key: str = ""
    cerebras_model: str = "llama-3.3-70b"
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "narratives"
    # Cosine distance threshold [0, 2] for narrative routing.
    # Stories with best_distance >= threshold spawn a new narrative direction.
    new_narrative_threshold: float = 0.40

    class Config:
        env_file = ".env"


settings = Settings()
