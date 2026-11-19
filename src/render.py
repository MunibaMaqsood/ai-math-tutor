"""Show AI text that contains LaTeX. Block equations ($$...$$) are drawn with
st.latex, and normal text with inline math ($...$) is drawn with st.markdown.
This avoids the red 'raw LaTeX' problem."""
import re

_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)


def split_math(text: str) -> list[tuple[str, str]]:
    """Return [('text', ...), ('math', ...), ...] in order."""
    parts, pos = [], 0
    for m in _DISPLAY.finditer(text):
        before = text[pos:m.start()].strip()
        if before:
            parts.append(("text", before))
        parts.append(("math", m.group(1).strip()))
        pos = m.end()
    rest = text[pos:].strip()
    if rest:
        parts.append(("text", rest))
    return parts


def render_math(text: str) -> None:
    import streamlit as st
    for kind, content in split_math(text):
        if kind == "math":
            st.latex(content)
        else:
            st.markdown(content)
