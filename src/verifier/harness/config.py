"""All runtime configuration, read from environment / .env (prefix GV_)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GV_", env_file=".env", extra="ignore")

    # keys (no prefix: standard names)
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    typesafe_api_key: str | None = Field(default=None, alias="TYPESAFE_API_KEY")
    dashscope_api_key: str | None = Field(default=None, alias="DASHSCOPE_API_KEY")

    # models
    llm_provider: str = "openai"  # openai | qwen (which generative model proposes structure)
    openai_model: str = "gpt-6-astra"
    qwen_model: str = "qwen-plus"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_schema_mode: str = "json_schema"  # json_schema (Max/Plus/Flash) | json_object (older)
    jev_model: str = "jev-latest"

    # input limits (chars; checked before any model call)
    max_answer_chars: int = 8_000
    max_passage_chars: int = 8_000
    max_passages: int = 20
    max_total_context_chars: int = 40_000

    # decision thresholds (tuned on the dev set; defaults are starting points)
    verdict_min_prob: float = 0.55     # top verdict option below this -> UNSUPPORTED
    relevance_min_prob: float = 0.60   # source sentence counts as evidence above this
    evidence_floor_prob: float = 0.30  # rank-fallback evidence must at least reach this
    fidelity_min_prob: float = 0.60    # decomposition unit accepted above this
    max_evidence_sentences: int = 4
    max_candidate_sentences: int = 60  # lexical prefilter above this many source sentences

    # batching / retries
    max_questions_per_request: int = 48
    max_parallel_requests: int = 4
    upstream_attempts: int = 3
    jev_timeout_s: float = 30.0
    openai_timeout_s: float = 90.0

    # storage
    db_path: str = "var/verifier.sqlite3"
    cache_mode: str = "off"  # off | record | replay
