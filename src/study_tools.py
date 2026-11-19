"""Stage 3: quizzes, flashcards, summaries and formula library from your PDFs.

Every tool works the same way: (1) retrieve the chunks that match a topic,
(2) ask Gemini to produce structured output based ONLY on those chunks.
Structured output (Pydantic) avoids broken JSON, even with LaTeX backslashes.
"""
from pydantic import BaseModel, Field

DEFAULT_TOPIC = "important definitions, theorems and formulas"

RULES = """Rules:
- Use ONLY the context below. Do not use outside knowledge.
- Write math in LaTeX with inline $...$ only (never $$).
- If the context is too small for the requested number, return fewer items.
- Set page to the page number from the [file, p. N] label, or 0 if unsure."""


# ---------------------------------------------------------------- data models
class MCQ(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly four choices, no A/B/C/D labels")
    answer_index: int = Field(description="0-based index of the correct option")
    explanation: str
    page: int = 0


class MCQSet(BaseModel):
    items: list[MCQ]


class WrittenQuestion(BaseModel):
    question: str
    model_answer: str
    page: int = 0


class WrittenSet(BaseModel):
    items: list[WrittenQuestion]


class Flashcard(BaseModel):
    front: str = Field(description="A term, question or formula name")
    back: str = Field(description="Short answer or definition")
    page: int = 0


class FlashcardSet(BaseModel):
    items: list[Flashcard]


class Formula(BaseModel):
    name: str
    latex: str = Field(description="The formula in LaTeX WITHOUT $ delimiters")
    meaning: str = Field(description="One or two sentences: what it means and when to use it")
    page: int = 0


class FormulaSet(BaseModel):
    items: list[Formula]


# ---------------------------------------------------------------- helpers
def get_context(topic: str, source: str | None = None, k: int = 10) -> str:
    """Retrieve the best-matching chunks (optionally from one document only)."""
    from src.vector_store import get_vector_store
    kwargs = {"filter": {"source": source}} if source else {}
    docs = get_vector_store().similarity_search(topic.strip() or DEFAULT_TOPIC, k=k, **kwargs)
    if not docs:
        raise ValueError("No matching content found. Upload and index a PDF first.")
    return "\n\n".join(
        f"[{d.metadata.get('source')}, p. {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


def strip_dollars(latex: str) -> str:
    """The AI sometimes wraps formulas in $...$ even when told not to."""
    return latex.strip().strip("$").strip()


def clean_mcqs(items: list) -> list:
    """Drop malformed questions (need exactly 4 options and a valid answer)."""
    return [q for q in items if len(q.options) == 4 and 0 <= q.answer_index < 4]


def _run_structured(schema, template: str, **kwargs):
    from langchain_core.prompts import ChatPromptTemplate
    from src.llm import get_llm
    chain = ChatPromptTemplate.from_template(template) | get_llm(0.3).with_structured_output(schema)
    last_error = None
    for _ in range(2):  # one retry, models occasionally return nothing
        try:
            result = chain.invoke(kwargs)
            if result is not None:
                return result
        except Exception as e:  # noqa: BLE001
            last_error = e 
        break  # quota used up: retrying only wastes requests
    raise RuntimeError(f"The AI did not return a valid result. {last_error or ''}")


# ---------------------------------------------------------------- tools
def make_mcqs(topic: str, source: str | None, n: int, difficulty: str) -> list[MCQ]:
    context = get_context(topic, source)
    result = _run_structured(
        MCQSet,
        """You are a mathematics teacher. Write {n} multiple-choice questions
({difficulty} difficulty) about: {topic}
Each question needs four plausible options, exactly one correct, and a short
explanation of why it is correct.
""" + RULES + "\n\nContext:\n{context}",
        n=n, difficulty=difficulty, topic=topic or DEFAULT_TOPIC, context=context)
    return clean_mcqs(result.items)


def make_written(topic: str, source: str | None, n: int, kind: str) -> list[WrittenQuestion]:
    context = get_context(topic, source)
    style = ("short-answer questions (1-3 sentence or one-line answers)" if kind == "short"
             else "long-answer questions that need a worked, multi-step answer")
    result = _run_structured(
        WrittenSet,
        """You are a mathematics teacher. Write {n} {style} about: {topic}
Give a complete model answer for each.
""" + RULES + "\n\nContext:\n{context}",
        n=n, style=style, topic=topic or DEFAULT_TOPIC, context=context)
    return result.items


def make_flashcards(topic: str, source: str | None, n: int) -> list[Flashcard]:
    context = get_context(topic, source)
    result = _run_structured(
        FlashcardSet,
        """Create {n} flashcards about: {topic}
Front: a term, question or formula name. Back: a short, clear answer.
""" + RULES + "\n\nContext:\n{context}",
        n=n, topic=topic or DEFAULT_TOPIC, context=context)
    return result.items


def find_formulas(topic: str, source: str | None, n: int = 10) -> list[Formula]:
    context = get_context(topic, source, k=12)
    result = _run_structured(
        FormulaSet,
        """List up to {n} important formulas, theorems and definitions about: {topic}
For definitions without a formula, write the key statement in LaTeX or \\text{{...}}.
""" + RULES + "\n\nContext:\n{context}",
        n=n, topic=topic or DEFAULT_TOPIC, context=context)
    return result.items


def summarize(topic: str, source: str | None) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from src.llm import ask
    context = get_context(topic, source, k=12)
    prompt = ChatPromptTemplate.from_template(
        """You are a mathematics tutor. Summarise the material below about: {topic}
Structure:
## Overview
## Key ideas (bullet points)
## Important definitions and formulas
## Key takeaways
Write math in LaTeX: inline $...$, and display equations as $$...$$ on their own lines.
Use ONLY the context. Cite pages like (file.pdf, p. 3).

Context:
{context}"""
    )
    return ask(prompt, topic=topic or "the main ideas", context=context)
