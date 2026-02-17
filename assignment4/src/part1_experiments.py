import textwrap

from .config import load_settings
from .gemini_client import GenerationParams, GeminiLLM


PROMPT = textwrap.dedent(
    """
    You are a helpful assistant. Write 6 short bullet points giving practical advice on:
    "How to avoid overfitting when training a model on noisy crypto price data".
    Keep bullets concise and actionable.
    """
).strip()


def run() -> None:
    try:
        settings = load_settings()
        llm = GeminiLLM(api_key=settings.google_api_key, model=settings.gemini_model)
    except Exception as e:
        raise SystemExit(
            "Config error.\n\n"
            "- Create `.env` (copy `env.example` -> `.env`)\n"
            "- Set a valid `GOOGLE_API_KEY` from https://aistudio.google.com/app/apikey\n"
            "- Optionally set `GEMINI_MODEL`\n\n"
            f"Details: {e}"
        ) from e

    experiments = [
        ("deterministic-ish", GenerationParams(temperature=0.2, top_p=0.9, top_k=10)),
        ("balanced", GenerationParams(temperature=0.7, top_p=0.95, top_k=40)),
        ("creative", GenerationParams(temperature=1.2, top_p=0.98, top_k=80)),
        ("high-top_p-low-top_k", GenerationParams(temperature=0.9, top_p=0.99, top_k=5)),
    ]

    for name, params in experiments:
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {name}")
        print(f"params={params}")
        print("-" * 80)
        try:
            out = llm.generate_text(prompt=PROMPT, params=params)
            print(out.strip())
        except Exception as e:
            raise SystemExit(
                "Gemini API call failed.\n\n"
                "- Double-check `GOOGLE_API_KEY` in `.env`\n"
                "- If this is a brand-new key, ensure you copied it correctly (no extra spaces)\n\n"
                f"Details: {e}"
            ) from e


if __name__ == "__main__":
    run()

