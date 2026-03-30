from flask import Flask, render_template, request, url_for
from sympy import symbols, Function, Eq, dsolve, Derivative, sin, cos, exp, tan, log, solve
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

# ================= PARSER =================
def parse_human_input(user_input):
    x = symbols('x')
    y = Function('y')

    s = user_input.strip()
    s = s.replace('\u2018', "'")
    s = s.replace('\u2019', "'")
    s = s.replace('\u2212', '-')
    s = s.replace("^", "**")

    # Derivadas
    s = re.sub(r"y\s*''", "Derivative(y(x), x, x)", s)
    s = re.sub(r"y\s*'",  "Derivative(y(x), x)",    s)

    # Multiplicaciones implícitas
    s = re.sub(r"(?P<num>\d+(?:\.\d+)?)\s*(?=[A-Za-z\(])", r"\g<num>*", s)
    s = re.sub(r"\)\s*(?=[A-Za-z0-9\(])", r")*", s)
    s = re.sub(r"(?<=[0-9\)])\s*(?=[A-Za-z\(])", r"*", s)

    # y → y(x)
    s = re.sub(r"\by\b(?!\s*\()", "y(x)", s)

    # Separar ecuación
    if '=' in s:
        lhs_s, rhs_s = s.split('=', 1)
    else:
        lhs_s, rhs_s = s, '0'

    local_dict = {
        'x': x, 'y': y,
        'sin': sin, 'cos': cos, 'exp': exp,
        'tan': tan, 'log': log,
        'Derivative': Derivative,
    }

    lhs = parse_expr(lhs_s, local_dict=local_dict)
    rhs = parse_expr(rhs_s, local_dict=local_dict)

    return Eq(lhs, rhs)


# ================= CONDICIONES INICIALES =================
def parse_initial_conditions(initials_text):
    if not initials_text:
        return None

    ic_regex = re.compile(
        r"y(?P<primes>'*)\((?P<x>[^)]+)\)\s*=\s*(?P<val>[-+]?\d*\.?\d+)"
    )
    matches = ic_regex.finditer(initials_text)

    ics = {}
    x0 = None

    for m in matches:
        order = len(m.group('primes'))
        x_val = float(m.group('x'))
        val   = float(m.group('val'))
        ics[order] = val
        x0 = x_val

    return (x0, ics) if ics else None


# ================= MÉTODO DE EULER =================
def euler_solve(f_system, x0, y0, x_end, n_steps=500):
    """
    Resuelve un sistema de ODEs de primer orden usando el Método de Euler.

    Fórmula:
        Y_{n+1} = Y_n + h * f(x_n, Y_n)

    Parámetros:
        f_system : función que recibe (x, Y) y devuelve dY/dx
        x0       : valor inicial de x
        y0       : vector de condiciones iniciales [y(x0), y'(x0), ...]
        x_end    : valor final de x
        n_steps  : número de pasos (más pasos = mayor precisión)

    Retorna:
        xs : array de valores de x
        ys : array de soluciones y(x)
    """
    h  = (x_end - x0) / n_steps          # tamaño de paso
    xs = np.zeros(n_steps + 1)
    ys = np.zeros((n_steps + 1, len(y0)))

    xs[0]  = x0
    ys[0]  = y0

    for i in range(n_steps):
        dY       = f_system(xs[i], ys[i])   # pendiente en el punto actual
        ys[i+1]  = ys[i] + h * dY           # paso de Euler
        xs[i+1]  = xs[i] + h

    return xs, ys


# ================= NUMÉRICO (con Euler) =================
def numeric_solve_and_plot(eq, initials_text=None, x_span=(0, 10)):
    x = symbols('x')
    y = Function('y')

    expr = sp.simplify(eq.lhs - eq.rhs)

    # Detectar orden máximo
    derivatives = list(expr.atoms(sp.Derivative))
    if not derivatives:
        raise ValueError(
            "No se detectaron derivadas. "
            "Escribe la ecuación usando y', y'' o y(x) y sus derivadas."
        )
    max_order = max(d.derivative_count for d in derivatives)

    # Despejar la derivada de mayor orden
    highest = [d for d in derivatives if d.derivative_count == max_order][0]
    sol_highest = solve(sp.Eq(expr, 0), highest)
    if not sol_highest:
        raise ValueError("No se pudo despejar la derivada principal de la ecuación.")
    rhs_expr = sol_highest[0]

    # Sustituir derivadas por variables auxiliares y0, y1, ..., y_{n-1}
    y_syms = [sp.Symbol(f'_y{i}') for i in range(max_order)]
    replacements = {
        sp.Derivative(y(x), (x, k)): y_syms[k]
        for k in range(max_order)
    }
    rhs_expr = rhs_expr.subs(replacements)

    # Convertir a función numérica
    f_rhs = sp.lambdify((x, *y_syms), rhs_expr, 'numpy')

    # Sistema de ecuaciones de primer orden:
    #   Y = [y, y', y'', ..., y^(n-1)]
    #   dY/dx = [y', y'', ..., y^(n-1), f_rhs(x, Y)]
    def f_system(xi, Yi):
        dY = np.zeros(max_order)
        for k in range(max_order - 1):
            dY[k] = Yi[k + 1]           # y_k' = y_{k+1}
        dY[-1] = f_rhs(xi, *Yi)         # y_{n-1}' = f(x, Y)
        return dY

    # Condiciones iniciales
    parsed = parse_initial_conditions(initials_text)
    if parsed:
        x0, ics = parsed
        y0 = [ics.get(i, 0.0) for i in range(max_order)]
    else:
        x0 = x_span[0]
        y0 = [1.0] + [0.0] * (max_order - 1)

    # ── Aplicar Método de Euler ──
    xs, ys = euler_solve(f_system, x0, y0, x_span[1], n_steps=500)

    # ── Gráfica ──
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#161622')
    ax.set_facecolor('#0e0e18')

    ax.plot(xs, ys[:, 0], color='#ff7a18', linewidth=2, label='y(x) — Euler')
    ax.set_xlabel('x', color='#aaaacc', fontsize=10)
    ax.set_ylabel('y(x)', color='#aaaacc', fontsize=10)
    ax.tick_params(colors='#5a5a78')
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a3e')
    ax.grid(True, color='#2a2a3e', linestyle='--', linewidth=0.6)
    ax.legend(facecolor='#1e1e2e', edgecolor='#2a2a3e', labelcolor='#dddde8', fontsize=9)

    plt.tight_layout()

    os.makedirs('static/plots', exist_ok=True)
    filename = f"plot_{int(time.time())}.png"
    path = f"static/plots/{filename}"
    plt.savefig(path, dpi=120)
    plt.close()

    return f"plots/{filename}"


# ================= ROUTE =================
@app.route('/', methods=['GET', 'POST'])
def index():
    solution       = None
    solution_latex = None
    error          = None
    plot_path      = None

    if request.method == 'POST':
        try:
            eq_input = request.form['equation']
            initials = request.form.get('initials', '')

            x_start_raw = request.form.get('x_start', '').strip()
            x_end_raw   = request.form.get('x_end',   '').strip()
            try:
                x_start = float(x_start_raw) if x_start_raw else 0.0
            except ValueError:
                x_start = 0.0
            try:
                x_end = float(x_end_raw) if x_end_raw else 10.0
            except ValueError:
                x_end = 10.0

            eq = parse_human_input(eq_input)

            # Intentar solución simbólica primero
            try:
                sol = dsolve(eq)
                solution = str(sol)
                try:
                    solution_latex = sp.latex(sol)
                except Exception:
                    solution_latex = None
            except Exception:
                # Si falla, usar Método de Euler
                plot = numeric_solve_and_plot(eq, initials, (x_start, x_end))
                plot_path = url_for('static', filename=plot)
                solution  = "Solución numérica por Método de Euler"

        except Exception as e:
            error = str(e)

    return render_template(
        'index.html',
        solution=solution,
        solution_latex=solution_latex,
        error=error,
        plot_path=plot_path
    )


if __name__ == '__main__':
    app.run()