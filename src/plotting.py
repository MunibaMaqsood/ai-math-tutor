"""Stage 2: plot one or more functions with Matplotlib."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from src.solver import parse


def plot_functions(exprs: list[str], var: str = "x", xmin: float = -10,
                   xmax: float = 10, ylim: tuple[float, float] | None = None):
    sym = sp.Symbol(var)
    xs = np.linspace(xmin, xmax, 1000)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for text in exprs:
        expr = parse(text)
        f = sp.lambdify(sym, expr, "numpy")
        with np.errstate(all="ignore"):
            ys = np.asarray(f(xs), dtype=complex) * np.ones_like(xs)
        ys = np.where(np.abs(ys.imag) < 1e-9, ys.real, np.nan)  # drop complex values
        ys[np.abs(ys) > 1e6] = np.nan
        ax.plot(xs, ys, label=f"${sp.latex(expr)}$")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.set_xlabel(var)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()
    return fig
