# MemLearn

An open-source agentic memory & continual learning system.

## Installation

1. Clone this repository and navigate to it
2. `python3.13 -m venv .venv` (at the moment, Python 3.14 doesn't work with
   ChromaDB)
3. `. .venv/bin/activate`
4. `uv sync --all-packages`

## Usage

Try it out locally with `uv run playground/main.py`.

You need to provide `OPENAI_API_KEY` and `COHERE_API_KEY` as environment
variables.
