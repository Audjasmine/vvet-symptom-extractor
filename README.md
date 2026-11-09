# V-Vet symptom extraction service

A small FastAPI service that turns a farmer's WhatsApp message about a sick animal into structured findings from a fixed list of signs, and decides in plain code whether the case needs escalating.

**Example**

`POST /extract` with `{"message": "There's foam coming from her nose and she can't get up."}` returns:

```json
{
  "findings": [
    {"sign_id": "froth_nose_mouth", "value": "yes", "quote": "There's foam coming from her nose"},
    {"sign_id": "unable_to_stand", "value": "yes", "quote": "she can't get up"}
  ],
  "escalate": true,
  "reason": "Urgent sign(s): froth_nose_mouth, unable_to_stand"
}
```

## How to run

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Create a `.env` file with a free Groq API key (https://console.groq.com/keys):

```
GROQ_API_KEY=your-key-here
```

Run the API:

```bash
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs to try `POST /extract`.

Run the tests (no internet or API key needed, the LLM is mocked):

```bash
python -m pytest -v
```

## Files

| File | What it does |
|---|---|
| `main.py` | The API: request/response models, validation of the LLM output, escalation rule, `/extract` endpoint |
| `llm.py` | The prompt and the call to the LLM (Groq, model `openai/gpt-oss-20b`) |
| `test_main.py` | 20 tests with a fake LLM |

## Decisions I made

**The LLM reads, the code decides.** The LLM's only job is turning free text into findings. Anything that affects safety is plain Python that can be tested: which signs are allowed, which findings to trust, and whether to escalate.

**Validating the LLM's output.** Every finding must pass these checks or it is dropped:
1. `sign_id` must be one of the 9 signs. I used a Pydantic `Literal` type, so anything else (e.g. `"fever"`) is rejected.
2. `value` must be `yes`, `no` or `not_sure`.
3. **The `quote` must really appear in the farmer's message.** This catches the LLM inventing evidence and supports the rule "only report signs the message actually mentions."
4. Each sign is reported at most once.

I drop bad findings one at a time instead of failing the whole request, so one bad item doesn't lose the good ones.

**Escalation is plain code.** `decide_escalation()` escalates when `froth_nose_mouth`, `unable_to_stand` or `cloudy_eye` is `yes` or `not_sure`. If the LLM adds its own `escalate` field, it is ignored (there is a test for this).

**Fail loudly, not silently.** If the LLM call fails or returns something that isn't valid JSON, the API returns **HTTP 502** instead of an empty result. An empty result would look like "nothing urgent", which is the dangerous mistake for a sick animal.

**Prompt.** It lists the signs with descriptions, explains when to use `not_sure` ("I think", "maybe"), says that "the animal is fine" is not a sign, and includes Shona hints (`haisi kudya` = not eating, `mbandabvu` = swelling) because farmers mix languages. Temperature is 0 and JSON mode is on, for consistent output.

**Provider.** I used Groq's free tier through its OpenAI-compatible API with a plain `httpx` call, so switching to another provider only means changing the URL and model name. While building, the model I first chose (`llama-3.3-70b-versatile`) had been retired and returned a 404, so I switched to `openai/gpt-oss-20b`. The service returned a 502 in that case, as designed.

## How I would measure extraction accuracy

1. **Build a labelled test set** of real (anonymised) farmer messages, with the correct signs and values marked by vets: English, Shona, mixed language, spelling mistakes, and messages with no signs.
2. **Run the real model over it** and measure for each sign:
   - **Precision:** of the signs we reported, how many were right?
   - **Recall:** of the signs that were really there, how many did we find?
   - **Value accuracy:** when we found the sign, was yes / no / not_sure correct?
3. **Escalation recall is the most important number.** Missing an urgent case is much worse than a false alarm.
4. Report results **per language** so poor Shona performance isn't hidden by the average.
5. Re-run this every time the prompt or model changes, to catch regressions.
6. In production, log the findings that get dropped by validation (bad sign or made-up quote) as a warning signal, and compare our results with the vet's final assessment.

## What I would do with more time

- Build the labelled evaluation set and a script that scores it automatically.
- Add a glossary of local farmer terms (Shona and Ndebele) for each sign, made with vets.
- Retry once if the LLM returns bad JSON, and use a strict JSON schema / tool calling where the provider supports it.
- Make the quote check more forgiving of small differences in punctuation or spacing.
- Handle messages about more than one animal, and the difference between past and current symptoms ("she wasn't eating last week but she's fine now").
- Split `main.py` into modules (models, validation, escalation) as it grows. Add logging, rate limiting and a fallback provider.

## How I used AI tools

I used Claude as an AI development assistant throughout the project. I started by understanding the problem and planning the application structure, then used Claude to explore possible approaches, clarify unfamiliar concepts and assist with implementing individual features.

I worked through the implementation step by step, running the application locally, reviewing the generated code and testing the functionality. When issues arose, I used AI assistance to investigate possible causes, understand the errors and identify solutions. For example, I encountered an issue when my initial Groq model was retired and returned a 404 error, which I resolved by switching to a supported model.

I also wrote and ran tests to verify the extraction and validation logic, including cases where the LLM returned invalid findings. I used AI as a support tool for learning, development and debugging, while taking responsibility for understanding the code and verifying that the application behaved as expected.