from __future__ import annotations

import argparse
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .config import load_settings
from .tools import calculator as _calculator_impl, get_weather as _get_weather_impl


@tool
def calculator(expression: str) -> str:
    """Safely evaluate a math expression (supports + - * / % **, sqrt, pi, etc.)."""
    return _calculator_impl(expression)


@tool
def get_weather(city: str) -> dict[str, Any]:
    """Get current weather for a city (Open-Meteo; no API key)."""
    return _get_weather_impl(city)


def build_agent(*, debug: bool = False):
    settings = load_settings()

    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,
    )

    tools = [calculator, get_weather]

    system_prompt = (
        "You are a helpful assistant. Use tools when needed. "
        "Keep answers concise and show final computed numbers clearly."
    )

    checkpointer = MemorySaver()
    return create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
        checkpointer=checkpointer,
        debug=debug,
        version="v2",
    )


def _last_ai_text(messages: list[Any]) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def run_demo(graph) -> None:
    """
    Non-interactive demo showing:
      - conversation memory (2nd question depends on the 1st)
      - tool calls (weather + calculator)
    """
    turns = [
        "What's the weather in Tbilisi right now?",
        "If the temperature increases by 3.5 degrees, what would it be?",
        "whats was the last question and the answer to it? Now what's the weather in Paris, and what's (12*7) + sqrt(81)?",
    ]
    cfg = {"configurable": {"thread_id": "demo"}}
    for t in turns:
        print("\n" + "=" * 80)
        print("USER:", t)
        out = graph.invoke({"messages": [HumanMessage(content=t)]}, config=cfg)  # pyright: ignore[reportArgumentType]
        print("ASSISTANT:", _last_ai_text(out["messages"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Part 4 (optional): LangChain/LangGraph agent (Gemini + tools + memory)")
    ap.add_argument("--demo", action="store_true", help="Run a scripted multi-turn demo (recommended).")
    ap.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False, help="Enable LangGraph debug logs.")
    args = ap.parse_args()

    graph = build_agent(debug=args.debug)

    if args.demo:
        run_demo(graph)
        return

    cfg = {"configurable": {"thread_id": "chat"}}
    while True:
        q = input("You (type 'quit' to exit): ").strip()
        if not q:
            continue
        if q.lower() in {"quit", "exit"}:
            break
        out = graph.invoke({"messages": [HumanMessage(content=q)]}, config=cfg)  # pyright: ignore[reportArgumentType]
        print("Assistant:", _last_ai_text(out["messages"]))


if __name__ == "__main__":
    main()

