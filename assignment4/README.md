## Week 4 — Part 1: LLM API Setup (Gemini)

This project implements **Part 1** of the Week 4 homework:
- LLM provider API setup (**Google Gemini**)
- Robust calling with **error handling**, **retries**, and **rate limiting**
- Simple script to **experiment with decoding parameters** (`temperature`, `top_k`, `top_p`)

It also includes **Part 2 (Structured Output)**: extracting a raw email into a validated Pydantic schema.

### Setup

Create a venv and install dependencies:

```bash
cd /Users/konstantine25b/Desktop/assignment4
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create your `.env` file (Cursor may block creating dotfiles automatically, so do it manually):

```bash
cp env.example .env
```

Edit `.env` and set:
- `GOOGLE_API_KEY`
- optionally `GEMINI_MODEL`

### Run the decoding experiments

```bash
source .venv/bin/activate
python -m src.part1_experiments
```

This prints multiple generations for the same prompt under different decoding settings so you can compare diversity vs. determinism.

### Part 2: Structured output (email extraction)

```bash
source .venv/bin/activate
python -m src.part2_email_structured
```

This:
- prompts the LLM to return **JSON-only**
- parses JSON
- validates with **Pydantic** (`ExtractedEmail`)
- if invalid, it re-prompts the LLM to **repair** the JSON and retries

### Part 3: Tool calling (calculator + weather)

Run the tool-calling agent:

```bash
source venv/bin/activate
python -m src.part3_tool_calling_agent "What's the weather in Tbilisi right now, and what is (12*7) + sqrt(81)?"
```

You should see debug logs showing:
- the model requesting tool calls
- the tool arguments
- tool outputs
- the model’s final answer after it receives tool results

### Part 4 (optional): LangChain integration (tools + memory)

Install dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

Run the LangChain agent demo (multi-turn, shows memory + tool calls):

```bash
python -m src.part4_langchain_agent --demo
```
