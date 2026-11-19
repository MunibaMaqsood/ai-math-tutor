import sympy as sp
from src.solver import (derivative, euclid_steps, integral, matrix_info,
                        modular, solve_equation, solve_system)


def test_quadratic():
    r = solve_equation("x^2 - 5x + 6 = 0")
    assert set(sp.sympify(r["plain"])) == {2, 3} and r["checked"]


def test_system():
    r = solve_system("2x + y = 5\nx - y = 1")
    assert "x: 2" in r["plain"] and "y: 1" in r["plain"] and r["checked"]


def test_derivative():
    r = derivative("x^3 + 2x", "x")
    assert sp.simplify(sp.sympify(r["plain"]) - (3 * sp.Symbol("x") ** 2 + 2)) == 0


def test_integral_checked():
    assert integral("2x")["checked"]
    assert integral("x^2", lower="0", upper="3")["plain"] == "9"


def test_matrix_det():
    assert "det=-2" in matrix_info("1 2; 3 4")["plain"]


def test_euclid():
    r = euclid_steps(48, 18)
    assert r["gcd"] == 6 and r["checked"] and r["steps"][0]["equation"] == "48 = 18 × 2 + 12"


def test_modular():
    r = modular(3, 7, 4)
    assert r["inverse"] == 5 and r["power"] == 4
