"""Shared Gemini helpers."""
from langchain_google_genai import ChatGoogleGenerativeAI

import config


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=temperature,
    )


def extract_text(content) -> str:
    """Gemini may return a string or a list of blocks; keep only the text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "".join(parts)


def ask(prompt_template, **kwargs) -> str:
    result = (prompt_template | get_llm()).invoke(kwargs)
    return extract_text(result.content)
