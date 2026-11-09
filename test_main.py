import json

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def fake_llm(response):
    """Return a fake call_llm that gives back `response` instead of calling Groq."""
    def _fake(message):
        if isinstance(response, Exception):
            raise response
        return json.dumps(response) if isinstance(response, dict) else response
    return _fake


def post(monkeypatch, llm_response, message):
    monkeypatch.setattr(main, "call_llm", fake_llm(llm_response))
    return client.post("/extract", json={"message": message})


# ---------- Escalation rule ----------

@pytest.mark.parametrize("sign", ["froth_nose_mouth", "unable_to_stand", "cloudy_eye"])
@pytest.mark.parametrize("value", ["yes", "not_sure"])
def test_escalates_on_urgent_signs(sign, value):
    escalate, _ = main.decide_escalation([main.Finding(sign_id=sign, value=value, quote="x")])
    assert escalate is True


def test_urgent_sign_with_no_does_not_escalate():
    escalate, _ = main.decide_escalation(
        [main.Finding(sign_id="froth_nose_mouth", value="no", quote="x")]
    )
    assert escalate is False


def test_non_urgent_signs_do_not_escalate():
    escalate, _ = main.decide_escalation(
        [main.Finding(sign_id="ticks", value="yes", quote="x"),
         main.Finding(sign_id="red_urine", value="yes", quote="x")]
    )
    assert escalate is False


# ---------- Sample messages through the API ----------

def test_sample_foam_and_cannot_stand(monkeypatch):
    msg = "There's foam coming from her nose and she can't get up."
    llm = {"findings": [
        {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "foam coming from her nose"},
        {"sign_id": "unable_to_stand", "value": "yes", "quote": "she can't get up"},
    ]}
    body = post(monkeypatch, llm, msg).json()
    assert len(body["findings"]) == 2
    assert body["escalate"] is True


def test_sample_unsure_cloudy_eye_escalates(monkeypatch):
    msg = "I think the eyes look a bit cloudy but I'm not sure. She has mbandabvu under the ear."
    llm = {"findings": [
        {"sign_id": "cloudy_eye", "value": "not_sure", "quote": "the eyes look a bit cloudy"},
        {"sign_id": "swelling", "value": "yes", "quote": "mbandabvu under the ear"},
    ]}
    body = post(monkeypatch, llm, msg).json()
    assert body["escalate"] is True


def test_sample_ticks_only_no_escalation(monkeypatch):
    msg = "The calf is fine, just lots of ticks on the ears."
    llm = {"findings": [{"sign_id": "ticks", "value": "yes", "quote": "lots of ticks"}]}
    body = post(monkeypatch, llm, msg).json()
    assert [f["sign_id"] for f in body["findings"]] == ["ticks"]
    assert body["escalate"] is False


def test_sample_shona_message(monkeypatch):
    msg = "Mombe yangu haisi kudya, and her urine is dark like tea."
    llm = {"findings": [
        {"sign_id": "reduced_appetite", "value": "yes", "quote": "haisi kudya"},
        {"sign_id": "red_urine", "value": "yes", "quote": "urine is dark like tea"},
    ]}
    body = post(monkeypatch, llm, msg).json()
    assert len(body["findings"]) == 2
    assert body["escalate"] is False


# ---------- Validation of LLM output ----------

def test_unknown_sign_id_is_dropped(monkeypatch):
    llm = {"findings": [{"sign_id": "fever", "value": "yes", "quote": "hot"}]}
    body = post(monkeypatch, llm, "She is hot").json()
    assert body["findings"] == []


def test_invalid_value_is_dropped(monkeypatch):
    llm = {"findings": [{"sign_id": "ticks", "value": "maybe", "quote": "ticks"}]}
    body = post(monkeypatch, llm, "I see ticks").json()
    assert body["findings"] == []


def test_made_up_quote_is_dropped(monkeypatch):
    llm = {"findings": [{"sign_id": "froth_nose_mouth", "value": "yes", "quote": "frothing badly"}]}
    body = post(monkeypatch, llm, "She has ticks").json()
    assert body["findings"] == []
    assert body["escalate"] is False


def test_duplicate_sign_kept_once(monkeypatch):
    llm = {"findings": [
        {"sign_id": "ticks", "value": "yes", "quote": "ticks"},
        {"sign_id": "ticks", "value": "yes", "quote": "ticks"},
    ]}
    body = post(monkeypatch, llm, "lots of ticks").json()
    assert len(body["findings"]) == 1


def test_llm_cannot_force_escalation(monkeypatch):
    llm = {"findings": [{"sign_id": "ticks", "value": "yes", "quote": "ticks"}],
           "escalate": True}
    body = post(monkeypatch, llm, "lots of ticks").json()
    assert body["escalate"] is False


# ---------- Failures ----------

def test_bad_json_from_llm_returns_502(monkeypatch):
    assert post(monkeypatch, "not json at all", "my cow is sick").status_code == 502


def test_llm_error_returns_502(monkeypatch):
    assert post(monkeypatch, RuntimeError("timeout"), "my cow is sick").status_code == 502


def test_missing_message_returns_422():
    assert client.post("/extract", json={}).status_code == 422