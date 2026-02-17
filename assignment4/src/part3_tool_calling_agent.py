from __future__ import annotations

import argparse
from typing import Any

from google.genai import types

from .config import load_settings
from .gemini_client import GenerationParams, GeminiLLM
from .tools import ToolError, calculator, get_weather


SYSTEM = """\
You are a helpful assistant.
You may call tools when needed.
If the user question is simple, answer directly without tools.
"""


def _tool_declarations() -> list[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name="calculator",
            description="Safely evaluate a math expression. Use for arithmetic or math functions.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "expression": types.Schema(
                        type=types.Type.STRING,
                        description="Math expression, e.g. 'sqrt(9) + 2*5' or 'pi*2'.",
                    )
                },
                required=["expression"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_weather",
            description="Get current weather for a city (temperature C, wind km/h, weather code).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "city": types.Schema(type=types.Type.STRING, description="City name, e.g. 'Tbilisi' or 'Paris'.")
                },
                required=["city"],
            ),
        ),
    ]


def _execute_tool(name: str, args: dict[str, Any]) -> Any:
    if name == "calculator":
        return {"result": calculator(expression=str(args.get("expression", "")))}
    if name == "get_weather":
        return get_weather(city=str(args.get("city", "")))
    raise ToolError(f"Unknown tool '{name}'")


def run_agent(question: str, *, debug: bool = True, max_turns: int = 6) -> str:
    """
    Manual tool-calling loop:
      - send user question + tool declarations
      - if model requests tool calls, execute them
      - feed tool results back as function responses
      - stop when model returns normal text without tool calls
    """
    settings = load_settings()
    llm = GeminiLLM(api_key=settings.google_api_key, model=settings.gemini_model)

    # Conversation as a list of `types.Content`
    messages: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=question)]),
    ]

    tool = types.Tool(function_declarations=_tool_declarations())
    params = GenerationParams(temperature=0.2, top_p=0.9, top_k=20, max_output_tokens=1024)

    for turn in range(1, max_turns + 1):
        if debug:
            print(f"\n[agent] turn={turn} sending to model")

        resp = llm.generate_response(
            contents=messages,
            params=params,
            system_instruction=SYSTEM,
            extra_config={"tools": [tool]},
        )

        # If the model produced tool calls, execute them and continue.
        calls = getattr(resp, "function_calls", None) or []
        if calls:
            if debug:
                print(f"[agent] model requested {len(calls)} tool call(s)")

            # IMPORTANT:
            # Append the *original* model content that contains the functionCall parts,
            # because Gemini requires provider-added `thought_signature` fields for tool use.
            model_content = getattr(getattr(resp, "candidates", [None])[0], "content", None)
            if model_content is None:
                # Fallback (should be rare): build parts from calls (may fail on some models).
                model_content = types.Content(
                    role="model",
                    parts=[types.Part.from_function_call(name=c.name, args=dict(c.args or {})) for c in calls],
                )
            messages.append(model_content)

            # Execute calls and return function responses
            response_parts: list[types.Part] = []
            for c in calls:
                try:
                    tool_result = _execute_tool(c.name, dict(c.args or {}))
                except Exception as e:
                    tool_result = {"error": str(e)}

                if debug:
                    print(f"[tool] {c.name} args={c.args} -> {tool_result}")

                response_parts.append(
                    types.Part.from_function_response(name=c.name, response=tool_result)
                )

            messages.append(types.Content(role="user", parts=response_parts))
            continue

        # No tool calls: return final text
        final_text = getattr(resp, "text", None) or ""
        if debug:
            print("[agent] final answer:")
            print(final_text.strip())
        return final_text.strip()

    raise RuntimeError(f"Agent exceeded max_turns={max_turns} without producing a final answer.")


def _render_prompt(system: str, messages: list[types.Content]) -> str:
    """
    Fallback string rendering for the wrapper call (not used for actual tool calling).
    Kept for readability in debug logs.
    """
    lines = [f"SYSTEM:\n{system}\n"]
    for m in messages:
        role = getattr(m, "role", "unknown")
        parts = getattr(m, "parts", []) or []
        text_parts = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                text_parts.append(t)
        if text_parts:
            lines.append(f"{role.upper()}:\n" + "\n".join(text_parts))
    return "\n\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Part 3: tool calling agent (calculator + weather)")
    ap.add_argument("question", type=str, help="User question, e.g. \"What's the weather in Tbilisi and what's 12*7?\"")
    ap.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    answer = run_agent(args.question, debug=args.debug)
    if not args.debug:
        print(answer)


if __name__ == "__main__":
    main()

