# Multi-Agent Orchestration System

A supervisor-specialist agent system built with LangGraph. A planner breaks a task into dependency-ordered subtasks, tool-using specialist agents execute them, and a reviewer checks every output before it moves on. Work in progress.

## What works today

- **Planner:** turns a task into a validated plan (Pydantic structured output, with a JSON fallback for small models). A code-level guardrail caps confidence when the request has blocking questions.
- **Clarification loop:** if the request is too vague, the graph asks the user questions and re-plans, with a hard cap of 2 rounds.
- **Tool registry:** every tool call passes through one gate that checks per-specialist permissions and rate limits, logs inputs, outputs and latency, and returns failures as data instead of raising. Includes a safe calculator (no eval) and sandboxed file tools.
- **RAG tool:** local Hugging Face embeddings (all-MiniLM-L6-v2) and ChromaDB, with a relevance cutoff chosen from measured distance distributions.
- **Specialist agents:** research, analysis, writing and code, each with a role prompt and only the tools it is allowed to use.
- **Executor:** runs subtasks in dependency order, passes outputs forward, retries once, and skips steps whose dependencies did not finish.
- **Reviewer:** rule-based checks (empty or raw-JSON output, error text, missing retrieval, numbers not found in the sources). A rejected output is sent back once with feedback.
- **Provider-agnostic LLM layer:** any OpenAI-compatible endpoint (Ollama, Hugging Face router, OpenAI, and others).

## Architecture

    START -> planner -> (blocking questions?) -> clarify -> planner   (max 2 rounds)
                     -> executor -> END

    executor: for each ready subtask (dependencies done)
        specialist agent -> tool registry -> tools (RAG search, calculator, files)
        reviewer -> approve | revise once with feedback | rejected / failed
        dependents of unfinished steps are skipped

## Run it

Requires Python 3.12 and [Ollama](https://ollama.com) (or any OpenAI-compatible endpoint).

    pip install -r requirements.txt
    ollama pull llama3.2:3b
    python make_sample_docs.py
    python test_rag.py
    python graph.py

Copy `.env.example` to `.env` and adjust it for your provider.

## Tests

    pytest -v

The reviewer, executor (with scripted fake specialists, so no LLM is needed) and registry are covered.

## Lessons learned while building

- A model's self-reported confidence was almost always 0.8, so the planner asks for concrete blocking questions and code decides the cap.
- Retrieval distance thresholds must be measured, not guessed. Short keyword queries score worse than full questions.
- Smaller chunks removed topic mixing in retrieved context.
- "Done" only meant "no exception", which is why the reviewer exists.
- Prompts must never mention tools an agent does not have.

## Roadmap

- Working memory and long-term semantic memory
- Human-in-the-loop approval (LangGraph interrupts)
- Tracing and a trace explorer
- LLM-based quality judge
- Docker and a review UI