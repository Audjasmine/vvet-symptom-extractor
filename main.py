import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from llm import call_llm

app = FastAPI()

SignId = Literal[
    "reduced_appetite",
    "weakness",
    "swelling",
    "rapid_breathing",
    "cloudy_eye",
    "froth_nose_mouth",
    "red_urine",
    "unable_to_stand",
    "ticks",
]

Value = Literal["yes", "no", "not_sure"]


class ExtractRequest(BaseModel):
    message: str


class Finding(BaseModel):
    sign_id: SignId
    value: Value
    quote: str


class ExtractResponse(BaseModel):
    findings: list[Finding]
    escalate: bool
    reason: str


# ---------- Escalation (plain code, not the LLM) ----------

URGENT_SIGNS = {"froth_nose_mouth", "unable_to_stand", "cloudy_eye"}


def decide_escalation(findings: list[Finding]) -> tuple[bool, str]:
    urgent = [
        f.sign_id
        for f in findings
        if f.sign_id in URGENT_SIGNS and f.value in ("yes", "not_sure")
    ]
    if urgent:
        return True, "Urgent sign(s): " + ", ".join(urgent)
    if findings:
        return False, "No urgent signs reported."
    return False, "No signs from the list were found in the message."


# ---------- Validation of the LLM output ----------

def validate_findings(raw_findings: list, message: str) -> list[Finding]:
    """Keep only findings with a valid sign_id and value, whose quote is
    really in the message. Drop duplicates and anything else."""
    valid = []
    seen = set()
    message_lower = message.lower()

    for item in raw_findings:
        try:
            finding = Finding(**item)
        except (ValidationError, TypeError):
            continue  # unknown sign_id, bad value, or not a dict

        if finding.quote.lower().strip() not in message_lower:
            continue  # the LLM made up a quote

        if finding.sign_id in seen:
            continue  # duplicate sign

        seen.add(finding.sign_id)
        valid.append(finding)

    return valid


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest):
    try:
        raw = call_llm(request.message)
        data = json.loads(raw)
        raw_findings = data["findings"]
    except Exception as e:
        # Fail loudly: an empty answer would look like "not urgent"
        raise HTTPException(status_code=502, detail=f"Could not extract findings: {e}")

    findings = validate_findings(raw_findings, request.message)
    escalate, reason = decide_escalation(findings)
    return ExtractResponse(findings=findings, escalate=escalate, reason=reason)