import os

import httpx
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You extract livestock health signs from farmers' WhatsApp messages.
Messages may mix English and Shona, e.g. "haisi kudya" means "not eating",
and "mbandabvu" means a swelling under the ear, jaw or in front of the shoulder.

Allowed sign_id values:
- reduced_appetite: eating less, or has stopped eating or grazing
- weakness: dull, weak, lying down more than usual
- swelling: a lump below the ear, under the jaw, or in front of the shoulder (mbandabvu)
- rapid_breathing: breathing fast while resting
- cloudy_eye: the eye looks white, grey or milky
- froth_nose_mouth: foam or froth from the nose or mouth
- red_urine: urine that is red, brown, or the colour of strong tea
- unable_to_stand: cannot get up or stay standing
- ticks: ticks can be seen on the animal

Rules:
- Only report signs the message actually mentions. Do not guess.
- value is "yes" if present, "no" if the farmer says it is absent,
  "not_sure" if the farmer sounds unsure ("I think", "maybe", "not sure").
- "The animal is fine" is not a mention of any sign.
- quote must be copied exactly from the message, word for word.
- Report each sign at most once.

Reply with JSON only, in this shape:
{"findings": [{"sign_id": "...", "value": "yes", "quote": "..."}]}
If no signs are mentioned, reply {"findings": []}"""


def call_llm(message: str) -> str:
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={
                "model": "openai/gpt-oss-20b",
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ],
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]