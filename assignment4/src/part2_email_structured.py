import json
import textwrap
import time
import argparse
from typing import Any

from pydantic import ValidationError

from .config import load_settings
from .gemini_client import GenerationParams, GeminiLLM
from .schemas import ExtractedEmail


EMAIL_SAMPLE = textwrap.dedent(
    """
    Subject: Invoice issue + urgent follow-up
    From: Alex Chen <alex.chen@vendor-example.com>
    To: Konstantine <konstantine@example.com>
    Date: 2026-02-16 09:12:00

    Hi Konstantine,

    Quick heads-up: our invoice #18421 for $2,450 USD is still marked unpaid.
    Could you confirm payment status by Feb 18? If you already paid, please send the transaction reference.

    Also, can we schedule a 20-min call this week (Thu or Fri afternoon)?
    Here’s the invoice PDF: https://example.com/invoices/18421.pdf

    Thanks,
    Alex
    """
).strip()


SYSTEM_INSTRUCTIONS = """\
You are an information extraction system.
Return ONLY valid JSON (no markdown, no commentary).
The JSON MUST exactly match the provided schema keys.
Use null for unknown optional fields; use [] for empty lists.
Dates must be ISO format YYYY-MM-DD. Datetimes must be ISO 8601.
URLs must be valid.
"""

TOP_LEVEL_KEYS = [
    "subject",
    "sent_at",
    "from",
    "to",
    "cc",
    "summary",
    "intent",
    "urgency",
    "sentiment",
    "mentioned_dates",
    "mentioned_money",
    "mentioned_links",
    "action_items",
]

INTENT_VALUES = {"request", "complaint", "support", "scheduling", "payment", "information", "other"}
URGENCY_VALUES = {"low", "medium", "high"}
SENTIMENT_VALUES = {"positive", "neutral", "negative", "mixed"}


def _schema_keys_hint() -> str:
    # We don't have JSON-schema enforcement available across all providers,
    # so we provide a strict key whitelist.
    return json.dumps(
        {
            "subject": "string|null",
            "sent_at": "datetime|null",
            "from": {"name": "string|null", "email": "string|null"},  # or null
            "to": [{"name": "string|null", "email": "string|null"}],
            "cc": [{"name": "string|null", "email": "string|null"}],
            "summary": "string",
            "intent": "request|complaint|support|scheduling|payment|information|other",
            "urgency": "low|medium|high",
            "sentiment": "positive|neutral|negative|mixed",
            "mentioned_dates": ["date"],
            "mentioned_money": [{"currency": "string", "amount": "number"}],
            "mentioned_links": ["url"],
            "action_items": [{"description": "string", "owner": "sender|recipient|unknown", "due_date": "date|null"}],
        },
        indent=2,
    )


def _preview(s: str, max_chars: int = 700) -> str:
    s = s.strip()
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _log(debug: bool, title: str, body: str | None = None) -> None:
    if not debug:
        return
    print(f"\n[step] {title}")
    if body is not None and body.strip():
        print(body)


def _extract_json(text: str) -> Any:
    """
    Best-effort extraction if the model surrounds JSON with stray text.
    """
    text = text.strip()
    # Fast path
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Use JSONDecoder raw parsing from the first '{'
    start = text.find("{")
    if start != -1:
        dec = json.JSONDecoder()
        obj, _end = dec.raw_decode(text[start:])
        return obj

    return json.loads(text)  # will raise (no object found)


def _normalize_for_schema(data: dict[str, Any]) -> dict[str, Any]:
    """
    Small, transparent normalization step so validation is reliable:
    - force `intent`, `urgency`, `sentiment` to the allowed lowercase literals when possible
    """
    out = dict(data)

    def _norm_literal(val: Any) -> str | None:
        if val is None:
            return None
        if not isinstance(val, str):
            return None
        return val.strip().lower()

    intent = _norm_literal(out.get("intent"))
    urgency = _norm_literal(out.get("urgency"))
    sentiment = _norm_literal(out.get("sentiment"))

    # Direct lowercase normalization
    if intent is not None:
        out["intent"] = intent
    if urgency is not None:
        out["urgency"] = urgency
    if sentiment is not None:
        out["sentiment"] = sentiment

    # Heuristic mapping if model outputs a sentence instead of a label
    if isinstance(out.get("intent"), str) and out["intent"] not in INTENT_VALUES:
        s = out["intent"]
        if any(k in s for k in ("invoice", "payment", "unpaid", "paid", "refund", "charge")):
            out["intent"] = "payment"
        elif any(k in s for k in ("schedule", "call", "meeting", "calendar", "thu", "fri")):
            out["intent"] = "scheduling"
        elif any(k in s for k in ("complaint", "angry", "unacceptable")):
            out["intent"] = "complaint"
        elif any(k in s for k in ("support", "help", "issue", "bug", "error")):
            out["intent"] = "support"
        elif any(k in s for k in ("request", "please", "could you", "can you")):
            out["intent"] = "request"
        else:
            out["intent"] = "other"

    if isinstance(out.get("urgency"), str) and out["urgency"] not in URGENCY_VALUES:
        s = out["urgency"]
        if "high" in s or "urgent" in s:
            out["urgency"] = "high"
        elif "medium" in s:
            out["urgency"] = "medium"
        elif "low" in s:
            out["urgency"] = "low"

    if isinstance(out.get("sentiment"), str) and out["sentiment"] not in SENTIMENT_VALUES:
        s = out["sentiment"]
        if "positive" in s:
            out["sentiment"] = "positive"
        elif "neutral" in s:
            out["sentiment"] = "neutral"
        elif "negative" in s:
            out["sentiment"] = "negative"
        elif "mixed" in s:
            out["sentiment"] = "mixed"

    return out


def extract_email_structured(
    llm: GeminiLLM,
    raw_email: str,
    *,
    max_attempts: int = 2,
    debug: bool = True,
) -> ExtractedEmail:
    """
    Prompt -> JSON -> Pydantic validation.
    If invalid, ask the LLM to repair the JSON using the validation error message.
    """
    _log(debug, "1) Build schema hint (preview)", _preview(_schema_keys_hint(), 900))
    keys_line = ", ".join(TOP_LEVEL_KEYS)
    prompt = textwrap.dedent(
        f"""
        SCHEMA (key whitelist + types):
        {_schema_keys_hint()}

        REQUIRED TOP-LEVEL KEYS (must include ALL of these, no extras):
        {keys_line}

        ENUM CONSTRAINTS (must use one of these exact lowercase strings):
        intent: {sorted(INTENT_VALUES)}
        urgency: {sorted(URGENCY_VALUES)}
        sentiment: {sorted(SENTIMENT_VALUES)}

        RAW EMAIL:
        {raw_email}
        """
    ).strip()

    params = GenerationParams(temperature=0.2, top_p=0.9, top_k=10, max_output_tokens=1024)

    last_err: Exception | None = None
    last_text: str | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            _log(debug, f"2) Attempt {attempt}: prompt preview", _preview(SYSTEM_INSTRUCTIONS + "\n\n" + prompt))
            _log(debug, f"3) Attempt {attempt}: call LLM", f"params={params} (JSON mode)")
            t0 = time.perf_counter()
            text = llm.generate_text(
                prompt=SYSTEM_INSTRUCTIONS + "\n\n" + prompt,
                params=params,
                extra_config={
                    "response_mime_type": "application/json",
                    # Strong structured output: SDK accepts a Pydantic model type here.
                    "response_schema": ExtractedEmail,
                },
            )
            dt = time.perf_counter() - t0
            last_text = text
            _log(debug, f"4) Attempt {attempt}: raw model output preview", _preview(text))
            _log(debug, f"5) Attempt {attempt}: parse JSON", "")
            data = _extract_json(text)
            if not isinstance(data, dict):
                raise ValueError("Parsed JSON is not an object/dict at the top level.")

            # Enforce exact key set early (before Pydantic) so repairs are targeted.
            got_keys = set(data.keys())
            expected_keys = set(TOP_LEVEL_KEYS)
            _log(debug, f"6) Attempt {attempt}: parsed JSON keys", ", ".join(sorted(list(got_keys))))
            if got_keys != expected_keys:
                missing = sorted(list(expected_keys - got_keys))
                extra = sorted(list(got_keys - expected_keys))
                raise ValueError(f"Top-level keys mismatch. missing={missing} extra={extra}")

            data = _normalize_for_schema(data)
            _log(debug, f"7) Attempt {attempt}: validate with Pydantic", f"LLM latency: {dt:.2f}s")
            model = ExtractedEmail.model_validate(data)
            _log(debug, f"8) Attempt {attempt}: success", "Structured output validated.")
            return model
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            last_err = e
            _log(debug, f"X) Attempt {attempt}: parse/validation failed", str(e))
            if attempt >= max_attempts:
                break

            # Repair prompt: show the invalid output + the error, ask for corrected JSON only.
            repair_prompt = textwrap.dedent(
                f"""
                The previous JSON did not parse/validate.
                Fix it and return ONLY corrected JSON (no commentary).

                REQUIRED TOP-LEVEL KEYS (must include ALL of these, no extras):
                {keys_line}

                ENUM CONSTRAINTS (must use one of these exact lowercase strings):
                intent: {sorted(INTENT_VALUES)}
                urgency: {sorted(URGENCY_VALUES)}
                sentiment: {sorted(SENTIMENT_VALUES)}

                SCHEMA (key whitelist + types):
                {_schema_keys_hint()}

                Validation/parse error:
                {str(e)}

                Previous output:
                {last_text}
                """
            ).strip()
            _log(debug, f"9) Attempt {attempt}: repair prompt preview", _preview(repair_prompt))
            prompt = repair_prompt
            params = GenerationParams(temperature=0.0, top_p=1.0, top_k=1, max_output_tokens=1024)

    raise RuntimeError(f"Failed to produce valid structured output after {max_attempts} attempts. Last error: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 2: Structured output email extraction")
    parser.add_argument("--email", type=str, default=None, help="Optional path to a .txt file containing a raw email.")
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print step-by-step intermediate outputs.",
    )
    args = parser.parse_args()

    settings = load_settings()
    llm = GeminiLLM(api_key=settings.google_api_key, model=settings.gemini_model)

    raw_email = EMAIL_SAMPLE
    if args.email:
        with open(args.email, "r", encoding="utf-8") as f:
            raw_email = f.read().strip()

    extracted = extract_email_structured(llm, raw_email, debug=args.debug)

    # Print pretty JSON (uses aliases, so `from_` becomes `from`)
    print(extracted.model_dump_json(by_alias=True, indent=2))


if __name__ == "__main__":
    main()

