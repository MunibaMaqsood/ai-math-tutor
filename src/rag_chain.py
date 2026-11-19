"""Retrieve context, ask Gemini, return answer + sources."""
from langchain_core.prompts import ChatPromptTemplate

import config
from src.llm import ask
from src.vector_store import retrieve

PROMPT = ChatPromptTemplate.from_template(
    """You are a patient, accurate mathematics tutor.
Answer the student's question using ONLY the context below.
If the context does not contain the answer, say exactly:
"I couldn't find this in the uploaded materials." and stop.

Rules:
- Explain in simple language, then give the formal statement.
- Write all math in LaTeX: $...$ inline and $$...$$ for display equations.
- Cite sources like (file.pdf, p. 12) after the facts you use.
{style}

Context:
{context}

Question: {question}

Answer:"""
)


STYLE_NOTES = {
    "Brief": "- Keep the answer short: at most 4 sentences plus any key formula.",
    "Normal": "",
    "Detailed": "- Give a thorough explanation with a worked example when the context allows.",
}


def format_context(docs) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('source')}, p. {d.metadata.get('page')}]\n{d.page_content}"
        for d in docs
    )


def answer_question(question: str) -> tuple[str, list[dict]]:
    docs = retrieve(question, k=config.TOP_K)
    if not docs:
        return "I couldn't find this in the uploaded materials.", []
    style = STYLE_NOTES.get(getattr(config, "ANSWER_STYLE", "Normal"), "")
    text = ask(PROMPT, context=format_context(docs), question=question, style=style)
    sources = [
        {"source": d.metadata.get("source"), "page": d.metadata.get("page"),
         "snippet": d.page_content[:300]}
        for d in docs
    ]
    return text, sources
