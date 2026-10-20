"""All runtime configuration, read from environment / .env (prefix GV_)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GV_", env_file=".env", extra="ignore")

    # which of the assignment's three methods serves POST /verify
    method: Literal["optimized", "minimal_llm", "baseline"] = "optimized"
    # method 1 (baseline): claims whose best source sentence covers less than
    # this IDF-weighted share of their content words become UNSUPPORTED.
    # Deliberately an untuned round default; retune on the eval dev split.
    baseline_coverage_min: float = 0.5

    # keys (no prefix: standard names)
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    typesafe_api_key: str | None = Field(default=None, alias="TYPESAFE_API_KEY")
    dashscope_api_key: str | None = Field(default=None, alias="DASHSCOPE_API_KEY")

    # models
    proposer: str = "llm"  # llm (generative model proposes structure) | jev (code cut points + Jev)
    jev_cut_min_prob: float = 0.60  # jev proposer: accept a cut above this probability
    # (cut-vote averaging removed 2026-10-22: measured no flip reduction on
    # the boundary-riding candidates it targeted; see design log)
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
    grounding_min_prob: float = 0.50   # diagnostic part of a kept-whole unit counts as grounded above this
    evidence_selector: str = "probe"   # probe | hybrid (probe + embedding prior)
    embed_model: str = "text-embedding-v3"
    embed_base_url: str | None = None  # None -> qwen_base_url; point at any OpenAI-compatible /v1 (e.g. local)
    embed_top_k: int = 2               # hybrid: embedding-ranked sentences eligible to join evidence
    embed_floor_prob: float = 0.25     # hybrid: minimum probe score for an embedding-prior sentence
    # when hybrid runs: "always" | "multi_passage" (request-level pre-trigger,
    # concurrent with the judge round) | "thin_evidence" (post-judge rescue of
    # units with thin evidence) | "both" (pre-trigger OR rescue)
    embed_trigger: str = "both"
    embed_min_sentences: int = 10      # pre-trigger also fires at this many source sentences
    embed_thin_min_words: int = 10     # rescue: claim this long with <=1 evidence sentence counts as thin
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
