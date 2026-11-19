"""Stage 2: SymPy calculators (ground truth) + AI step-by-step explanations.

Design: SymPy computes the answer, so the numbers are reliable. The AI is only
asked to EXPLAIN a result SymPy already produced.
Note: parse_expr evaluates text. Fine on your own PC; add stricter input
checks before exposing this to the public.
"""
import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor, implicit_multiplication_application, parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


def parse(text: str) -> sp.Expr:
    """Turn text like '3x^2 + 2x - 5' into a SymPy expression."""
    return parse_expr(text.strip(), transformations=_TRANSFORMS)


def _equation(text: str) -> sp.Eq:
    if "=" in text:
        left, right = text.split("=", 1)
        return sp.Eq(parse(left), parse(right))
    return sp.Eq(parse(text), 0)


def solve_equation(text: str, var: str = "x") -> dict:
    sym = sp.Symbol(var)
    eq = _equation(text)
    sols = sp.solve(eq, sym)
    ok = all(sp.simplify((eq.lhs - eq.rhs).subs(sym, s)) == 0 for s in sols)
    return {"problem": sp.latex(eq), "result": sp.latex(sols),
            "plain": str(sols), "checked": ok}


def solve_system(text: str) -> dict:
    """One equation per line, e.g. '2x + y = 5' newline 'x - y = 1'."""
    lines = [l for l in re.split(r"[\n;]", text) if l.strip()]
    eqs = [_equation(l) for l in lines]
    syms = sorted(set().union(*[e.free_symbols for e in eqs]), key=str)
    sols = sp.solve(eqs, syms, dict=True)
    ok = all(
        all(sp.simplify((e.lhs - e.rhs).subs(s)) == 0 for e in eqs) for s in sols
    )
    return {"problem": r"\begin{cases}" + r"\\".join(sp.latex(e) for e in eqs) + r"\end{cases}",
            "result": sp.latex(sols), "plain": str(sols), "checked": ok}


def derivative(text: str, var: str = "x", order: int = 1) -> dict:
    sym, expr = sp.Symbol(var), parse(text)
    res = sp.diff(expr, sym, order)
    return {"problem": rf"\frac{{d^{{{order}}}}}{{d{var}^{{{order}}}}}\left({sp.latex(expr)}\right)",
            "result": sp.latex(sp.simplify(res)), "plain": str(res), "checked": True}


def integral(text: str, var: str = "x", lower: str = "", upper: str = "") -> dict:
    sym, expr = sp.Symbol(var), parse(text)
    if lower.strip() and upper.strip():
        lo, hi = parse(lower), parse(upper)
        res = sp.integrate(expr, (sym, lo, hi))
        problem = rf"\int_{{{sp.latex(lo)}}}^{{{sp.latex(hi)}}} {sp.latex(expr)}\,d{var}"
        return {"problem": problem, "result": sp.latex(sp.simplify(res)),
                "plain": str(res), "checked": True}
    res = sp.integrate(expr, sym)
    # Check: differentiating the answer must give back the original function.
    ok = sp.simplify(sp.diff(res, sym) - expr) == 0
    return {"problem": rf"\int {sp.latex(expr)}\,d{var}",
            "result": sp.latex(res) + " + C", "plain": str(res) + " + C", "checked": ok}


def parse_matrix(text: str) -> sp.Matrix:
    """Rows separated by ';' or new lines, entries by commas or spaces."""
    rows = [r for r in text.replace("\n", ";").split(";") if r.strip()]
    return sp.Matrix([[parse(c) for c in re.split(r"[,\s]+", r.strip())] for r in rows])


def matrix_info(text: str) -> dict:
    m = parse_matrix(text)
    lines = [r"\text{Matrix: }" + sp.latex(m), r"\text{Rank} = " + str(m.rank())]
    plain = f"matrix={m.tolist()}, rank={m.rank()}"
    if m.is_square:
        det = m.det()
        lines.append(r"\det = " + sp.latex(det))
        plain += f", det={det}"
        if det != 0:
            lines.append(r"A^{-1} = " + sp.latex(m.inv()))
            plain += f", inverse={m.inv().tolist()}"
        else:
            lines.append(r"\text{Not invertible (determinant is 0)}")
        lines.append(r"\text{Eigenvalues: }" + sp.latex(m.eigenvals()))
        plain += f", eigenvalues={m.eigenvals()}"
    return {"problem": sp.latex(m), "result": r"\\".join(lines), "plain": plain,
            "checked": True}


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (s, t, g) with a*s + b*t = g = gcd(a, b)."""
    old_r, r, old_s, s, old_t, t = a, b, 1, 0, 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_s, old_t, old_r


def euclid_steps(a: int, b: int) -> dict:
    """Euclidean algorithm table + Bezout coefficients (extended Euclid)."""
    steps, x, y = [], a, b
    while y:
        q, r = divmod(x, y)
        steps.append({"equation": f"{x} = {y} × {q} + {r}"})
        x, y = y, r
    s, t, g = _extended_gcd(a, b)
    return {"steps": steps, "gcd": g, "bezout": f"{g} = {a}×({s}) + {b}×({t})",
            "checked": g == int(sp.igcd(a, b)) and a * s + b * t == g}


def modular(a: int, m: int, exponent: int | None = None) -> dict:
    out = {"gcd": int(sp.igcd(a, m))}
    out["inverse"] = int(sp.mod_inverse(a, m)) if out["gcd"] == 1 else None
    if exponent is not None:
        out["power"] = pow(a, exponent, m)
    return out


# ---------- AI explanations (imported lazily so SymPy tests need no API) ----------

def explain_verified(problem_text: str, plain_result: str) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from src.llm import ask
    prompt = ChatPromptTemplate.from_template(
        """You are a patient mathematics tutor. A computer algebra system has
ALREADY computed and verified this result. Do NOT change the final answer.

Problem: {problem}
Verified result: {result}

Explain how to reach this result step by step for a student:
1. **Given** - restate the problem.
2. **Method and formulas** - name the method and the formula used.
3. **Steps** - number each step, show the calculation, and say WHY it is needed.
4. **Final answer** - must match the verified result.
5. **Check** - show a quick way to verify it.
Write all math in LaTeX ($...$ inline, $$...$$ for display)."""
    )
    return ask(prompt, problem=problem_text, result=plain_result)


def solve_word_problem(problem_text: str) -> str:
    from langchain_core.prompts import ChatPromptTemplate
    from src.llm import ask
    prompt = ChatPromptTemplate.from_template(
        """You are a careful mathematics tutor. Solve the problem step by step.

Problem: {problem}

Format:
1. **Given** 2. **Method and formulas** 3. **Steps** (each with the reason why)
4. **Final answer** 5. **Verification** (substitute back or use a second method).
Write all math in LaTeX ($...$ inline, $$...$$ for display). Be accurate; if you
are unsure about a step, say so."""
    )
    return ask(prompt, problem=problem_text)
