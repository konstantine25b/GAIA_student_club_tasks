from __future__ import annotations

import ast
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class ToolError(RuntimeError):
    pass


# =========================
# Calculator (safe parsing)
# =========================

_ALLOWED_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Load,
    ast.Call,
    ast.Name,
    ast.Constant,
)


def calculator(expression: str) -> str:
    """
    Safely evaluate a basic math expression.

    Supported:
      - +, -, *, /, %, **, parentheses
      - constants: pi, e
      - functions: sqrt, abs, round, floor, ceil, log, log10, exp, sin, cos, tan

    Returns a string (so it's easy to display and pass back to the LLM).
    """
    expr = (expression or "").strip()
    if not expr:
        raise ToolError("calculator: expression is empty")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"calculator: invalid syntax: {e}") from e

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ToolError(f"calculator: disallowed syntax: {type(node).__name__}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ToolError("calculator: only simple function calls are allowed (e.g., sqrt(9))")
            if node.func.id not in _ALLOWED_FUNCS:
                raise ToolError(f"calculator: unknown function '{node.func.id}'")

        if isinstance(node, ast.Name) and node.id not in _ALLOWED_FUNCS:
            raise ToolError(f"calculator: unknown identifier '{node.id}'")

    def _eval(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ToolError("calculator: only int/float constants are allowed")
        if isinstance(n, ast.Name):
            return _ALLOWED_FUNCS[n.id]
        if isinstance(n, ast.UnaryOp):
            v = _eval(n.operand)
            if isinstance(n.op, ast.UAdd):
                return +v
            if isinstance(n.op, ast.USub):
                return -v
            raise ToolError("calculator: unsupported unary operator")
        if isinstance(n, ast.BinOp):
            a = _eval(n.left)
            b = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if isinstance(n.op, ast.Div):
                return a / b
            if isinstance(n.op, ast.Mod):
                return a % b
            if isinstance(n.op, ast.Pow):
                return a ** b
            raise ToolError("calculator: unsupported binary operator")
        if isinstance(n, ast.Call):
            fn = _eval(n.func)
            args = [_eval(a) for a in n.args]
            return fn(*args)
        raise ToolError(f"calculator: unsupported node {type(n).__name__}")

    val = _eval(tree)
    if isinstance(val, float):
        if math.isinf(val) or math.isnan(val):
            raise ToolError("calculator: result is not a finite number")

    return str(val)


# =========================
# Weather (Open-Meteo)
# =========================


@dataclass(frozen=True)
class WeatherResult:
    city: str
    latitude: float
    longitude: float
    temperature_c: float
    wind_speed_kph: float
    weather_code: int


def _http_get_json(url: str, *, timeout_s: float = 10.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "assignment4-tool-calling/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def get_weather(city: str) -> dict[str, Any]:
    """
    Get current weather for a city using Open-Meteo (no API key required).

    Steps:
      1) geocode city -> lat/lon
      2) fetch current weather for those coordinates
    """
    q = (city or "").strip()
    if not q:
        raise ToolError("get_weather: city is empty")

    geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
        {"name": q, "count": 1, "language": "en", "format": "json"}
    )
    geo = _http_get_json(geo_url)
    results = geo.get("results") or []
    if not results:
        raise ToolError(f"get_weather: could not find city '{q}'")

    r0 = results[0]
    lat = float(r0["latitude"])
    lon = float(r0["longitude"])
    resolved_name = str(r0.get("name") or q)

    weather_url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,wind_speed_10m,weather_code",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "UTC",
        }
    )
    w = _http_get_json(weather_url)
    current = w.get("current") or {}
    if not current:
        raise ToolError("get_weather: missing current weather in response")

    out = WeatherResult(
        city=resolved_name,
        latitude=lat,
        longitude=lon,
        temperature_c=float(current["temperature_2m"]),
        wind_speed_kph=float(current["wind_speed_10m"]),
        weather_code=int(current["weather_code"]),
    )
    return {
        "city": out.city,
        "latitude": out.latitude,
        "longitude": out.longitude,
        "temperature_c": out.temperature_c,
        "wind_speed_kph": out.wind_speed_kph,
        "weather_code": out.weather_code,
    }

