"""Study-plan generation through Groq, with optional Tavily research context."""

from __future__ import annotations

import re
from collections.abc import Callable

from study_assistant.providers.groq import chat_completion
from study_assistant.providers.tavily import tavily_search

CURRENT_INFO_PATTERN = re.compile(
    r"\b(latest|current|currently|recent|up[- ]to[- ]date|this year|newest|recent research|latest research)\b",
    re.IGNORECASE,
)


def needs_current_research(question: str) -> bool:
    """Only search when the request explicitly needs current or recent information."""
    return bool(CURRENT_INFO_PATTERN.search(question))


def _research_context(config: dict[str, str], question: str, search: Callable[..., dict]) -> str:
    try:
        response = search(
            config,
            question[:1000],
            search_depth="advanced",
            max_results=4,
            include_answer=False,
            safe_search=True,
        )
    except (ValueError, TimeoutError):
        return "Tavily research was unavailable. Do not claim current facts; still make a useful general plan."

    results = response.get("results", []) if isinstance(response, dict) else []
    excerpts = []
    if isinstance(results, list):
        for item in results[:4]:
            if not isinstance(item, dict):
                continue
            title = item.get("title")
            content = item.get("content")
            if isinstance(title, str) and isinstance(content, str) and title.strip() and content.strip():
                excerpts.append(f"- {title.strip()[:180]}: {content.strip()[:700]}")
    if not excerpts:
        return "Tavily returned no useful current sources. Do not invent current facts; still make a useful general plan."
    return "Supplementary current research from Tavily (use only when relevant):\n" + "\n".join(excerpts)


def generate_plan_answer(
    prompt: str,
    question: str,
    config: dict[str, str],
    *,
    has_document: bool = False,
    search: Callable[..., dict] = tavily_search,
    complete: Callable[[dict[str, str], str], str] = chat_completion,
) -> str:
    """Generate a plan with Groq; use Tavily only for explicit current-info requests."""
    if needs_current_research(question):
        context = _research_context(config, question, search)
        prompt = f"{prompt}\n\n{context}"
    elif has_document:
        prompt += "\n\nDo not use web research; treat the learner's document as the primary reference."
    return complete(config, prompt)
