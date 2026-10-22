from fastapi.testclient import TestClient

from conftest import ANSWER, PASSAGE_0, PASSAGE_1, FakeLLM, scripted_decider
from verifier.harness.config import Settings
from verifier.harness.errors import MalformedModelOutput, UpstreamError
from verifier.harness.pipeline import Dependencies
from verifier.service import create_app


def make_client(llm, decider, **overrides) -> TestClient:
    # scripted proposals: the llm proposer entry
    settings = Settings(_env_file=None, openai_api_key="t", typesafe_api_key="t",
                        **{"proposer": "llm", "sentence_guard": False, **overrides})
    deps = Dependencies(settings=settings, llm=llm, decider=decider, conn=None)
    app = create_app(settings, dependencies=deps)
    return TestClient(app, raise_server_exceptions=False)


def test_verify_happy_path(wal_proposal, wal_decider):
    with make_client(FakeLLM(proposal=wal_proposal), wal_decider) as client:
        resp = client.post(
            "/verify",
            json={"context_passages": [PASSAGE_0, PASSAGE_1], "generated_answer": ANSWER},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["claims"]) == 4
    assert {c["verdict"] for c in body["claims"]} == {"SUPPORTED", "CONTRADICTED", "UNSUPPORTED"}


def test_health():
    with make_client(FakeLLM(), scripted_decider()) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_invalid_body_is_422():
    with make_client(FakeLLM(), scripted_decider()) as client:
        resp = client.post("/verify", json={"generated_answer": 7})
    assert resp.status_code == 422


def test_oversized_input_is_413(wal_decider):
    with make_client(FakeLLM(), wal_decider, max_answer_chars=10) as client:
        resp = client.post(
            "/verify", json={"context_passages": ["x"], "generated_answer": "a" * 50}
        )
    assert resp.status_code == 413
    assert resp.json()["error"] == "input_too_large"


def test_transient_upstream_error_is_503(wal_proposal):
    class FailingDecider:
        name = "fail"

        async def ask(self, state, questions):
            raise UpstreamError(provider="jev", kind="transient", retryable=True)

    with make_client(FakeLLM(proposal=wal_proposal), FailingDecider()) as client:
        resp = client.post(
            "/verify", json={"context_passages": [PASSAGE_0], "generated_answer": ANSWER}
        )
    assert resp.status_code == 503
    assert resp.json()["error"] == "upstream_error"


def test_fatal_upstream_error_is_502(wal_proposal):
    class FatalDecider:
        name = "fail"

        async def ask(self, state, questions):
            raise UpstreamError(provider="jev", kind="fatal", retryable=False)

    with make_client(FakeLLM(proposal=wal_proposal), FatalDecider()) as client:
        resp = client.post(
            "/verify", json={"context_passages": [PASSAGE_0], "generated_answer": ANSWER}
        )
    assert resp.status_code == 502


def test_malformed_decider_output_is_502(wal_proposal):
    class MalformedDecider:
        name = "bad"

        async def ask(self, state, questions):
            raise MalformedModelOutput("jev", "missing answers")

    with make_client(FakeLLM(proposal=wal_proposal), MalformedDecider()) as client:
        resp = client.post(
            "/verify", json={"context_passages": [PASSAGE_0], "generated_answer": ANSWER}
        )
    assert resp.status_code == 502
    assert resp.json()["error"] == "malformed_model_output"


def test_empty_answer_is_200_empty_claims(wal_decider):
    with make_client(FakeLLM(), wal_decider) as client:
        resp = client.post("/verify", json={"context_passages": [PASSAGE_0], "generated_answer": ""})
    assert resp.status_code == 200
    assert resp.json() == {"claims": []}


def test_llm_provider_selection():
    from verifier.service import llm_config

    qwen = llm_config(Settings(_env_file=None, llm_provider="qwen", dashscope_api_key="k"))
    assert qwen.name == "qwen" and qwen.api == "chat"
    assert "dashscope" in qwen.base_url
    assert qwen.extra_body == {"enable_thinking": False}

    openai_cfg = llm_config(Settings(_env_file=None, llm_provider="openai", openai_api_key="k"))
    assert openai_cfg.name == "openai" and openai_cfg.api == "responses"
    assert openai_cfg.base_url is None

    import pytest as _pytest

    with _pytest.raises(ValueError):
        llm_config(Settings(_env_file=None, llm_provider="nope"))

    # the reported method-2 configuration: gpt-6.1-sol, no temperature parameter
    assert openai_cfg.model == "gpt-6.1-sol" and openai_cfg.temperature is None
    zero = llm_config(Settings(_env_file=None, llm_provider="openai", openai_api_key="k", openai_temperature_zero=True))
    assert zero.temperature == 0.0


def test_default_proposer_is_jev():
    # the reported results use the jev proposer; the shipped default must match
    assert Settings(_env_file=None).proposer == "jev"
