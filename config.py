"""Central configuration for the memory optimization project."""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # --- LLM API ---
    llm_api_key: str = os.getenv("OPENAI_API_KEY", "sk-your-key-here")
    judge_api_key: str = os.getenv("JUDGE_API_KEY", "")
    llm_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
    generator_model: str = "deepseek-chat"
    judge_model: str = "deepseek-chat"
    llm_temperature: float = 0.8
    llm_max_tokens: int = 2048

    # --- Iteration ---
    num_rounds: int = 5
    candidates_per_round: int = 5
    compare_top_k: int = 2
    compare_bottom_k: int = 2

    # --- Experience Library ---
    experience_max_size: int = 10
    experience_top_n: int = 10
    dedup_threshold: float = 0.8
    rules_per_round_max: int = 3   # max new rules added per round (gradual growth)

    # --- scoring models ---
    nli_model: str = "local_models/roberta-large-mnli"
    embedding_model: str = "local_models/all-MiniLM-L6-v2"

    # --- Entity Extraction ---
    spacy_model: str = "en_core_web_sm"
    entity_fields: Optional[list] = None

    # --- Early Stopping ---
    early_stop_patience: int = 3
    early_stop_min_delta: float = 0.005

    # --- Paths ---
    data_dir: str = "data/samples"
    output_dir: str = "output"
    cache_dir: str = "cache"

    # --- Seed ---
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)


config = Config()
