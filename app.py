from flask import Flask, render_template, request, url_for
import sympy as sp
from sympy import symbols, Function, Eq, dsolve, Derivative, sin, cos, exp, tan, log, solve, sqrt, pi, E
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

# ============================================================
#  PARSER ULTRA-ROBUSTO
# ============================================================

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

def normalize_input(s: str) -> str:
    """Limpia y normaliza la entrada antes de parsear."""
    # Caracteres especiales tipográficos
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u2212', '-').replace('\u00b2', '**2').replace('\u00b3', '**3')
    s = s.replace('^', '**')
    s = s.strip()

    # Normalizar espacios alrededor de = y operadores
    s = re.sub(r'\s+', ' ', s)
    return s


def replace_derivatives(s: str) -> str:
    """
    Convierte múltiples notaciones de derivadas a Derivative(...).
    Soporta: y'', y', y''', d2y/dx2, d^2y/dx^2, dy/dx, etc.
    """
    # d^n y / dx^n  o  d(n)y/dx(n)
    s = re.sub(r"d\^?(\d+)\s*y\s*/\s*d[xt]\^?\1", lambda m: f"Derivative(y(x), x, {m.group(1)})", s)
    # dy/dx  o  dy/dt
    s = re.sub(r"dy\s*/\s*d[xt]", "Derivative(y(x), x)", s)
    # y''' (3 primes)
    s = re.sub(r"y\s*'''", "Derivative(y(x), x, 3)", s)
    # y'' (2 primes)
    s = re.sub(r"y\s*''", "Derivative(y(x), x, 2)", s)
    # y'  (1 prime)
    s = re.sub(r"y\s*'", "Derivative(y(x), x)", s)
    return s


def replace_functions(s: str) -> str:
    """Expande alias de funciones comunes."""
    aliases = {
        r'\bsen\b': 'sin',
        r'\bSen\b': 'sin',
        r'\bSin\b': 'sin',
        r'\bCos\b': 'cos',
        r'\bTan\b': 'tan',
        r'\bExp\b': 'exp',
        r'\bLn\b':  'log',
        r'\bln\b':  'log',
        r'\bLog\b': 'log',
        r'\bSqrt\b': 'sqrt',
        r'\bsqrt\b': 'sqrt',
        r'\bpi\b': 'pi',
        r'\bPI\b': 'pi',
    }
    for pat, repl in aliases.items():
        s = re.sub(pat, repl, s)
    return s


def fix_implicit_multiplication(s: str) -> str:
    """Agrega * donde falta (2y → 2*y, 3sin → 3*sin, etc.)."""
    # número seguido de letra o paréntesis
    s = re.sub(r'(\d+\.?\d*)\s*([a-zA-Z\(])', r'\1*\2', s)
    # ) seguido de letra o número o (
    s = re.sub(r'\)\s*([a-zA-Z0-9\(])', r')*\1', s)
    return s


def replace_standalone_y(s: str) -> str:
    """y sola (sin paréntesis ni prima) → y(x)."""
    return re.sub(r'\by\b(?!\s*[\(\'\'])', 'y(x)', s)


def parse_human_input(user_input: str) -> Eq:
    """
    Parser principal. Intenta múltiples estrategias para interpretar
    la ecuación diferencial escrita por el usuario.
    """
    x = symbols('x')
    y = Function('y')

    local_dict = {
        'x': x, 'y': y,
        'sin': sin, 'cos': cos, 'exp': exp,
        'tan': tan, 'log': log, 'sqrt': sqrt,
        'pi': pi, 'e': E,
        'Derivative': Derivative,
        'y': y,
    }

    s = normalize_input(user_input)
    s = replace_functions(s)
    s = replace_derivatives(s)
    s = fix_implicit_multiplication(s)
    s = replace_standalone_y(s)

    # Separar lhs = rhs
    if '=' in s:
        lhs_s, rhs_s = s.split('=', 1)
    else:
        lhs_s, rhs_s = s, '0'

    try:
        lhs = parse_expr(lhs_s, local_dict=local_dict, transformations=TRANSFORMATIONS)
        rhs = parse_expr(rhs_s, local_dict=local_dict, transformations=TRANSFORMATIONS)
        return Eq(lhs, rhs)
    except Exception as e:
        raise ValueError(f"No se pudo interpretar la ecuación: '{user_input}'\nError técnico: {e}")


# ============================================================
#  CONDICIONES INICIALES
# ============================================================

def parse_initial_conditions(text: str):
    if not text or not text.strip():
        return None

    ic_regex = re.compile(
        r"y(?P<primes>'*)\s*\(\s*(?P<x>[+-]?\d*\.?\d+)\s*\)\s*=\s*(?P<val>[+-]?\d*\.?\d+)"
    )
    matches = list(ic_regex.finditer(text))

    if not matches:
        # Intentar formato y(0)=0, y'(0)=1 sin espacios
        text2 = text.replace(' ', '')
        matches = list(ic_regex.finditer(text2))

    ics = {}
    x0 = None

    for m in matches:
        order = len(m.group('primes'))
        x_val = float(m.group('x'))
        val   = float(m.group('val'))
        ics[order] = val
        x0 = x_val

    return (x0, ics) if ics else None


# ============================================================
#  MÉTODO DE EULER
# ============================================================

def euler_solve(f_system, x0, y0, x_end, n_steps=1000):
    """
    Resuelve un sistema de ODEs con el Método de Euler explícito.

    Y_{n+1} = Y_n + h · f(x_n, Y_n)
    """
    h  = (x_end - x0) / n_steps
    n  = len(y0)
    xs = np.zeros(n_steps + 1)
    ys = np.zeros((n_steps + 1, n))

    xs[0] = x0
    ys[0] = y0

    for i in range(n_steps):
        try:
            dY = f_system(xs[i], ys[i])
            ys[i+1] = ys[i] + h * np.array(dY)
        except Exception:
            ys[i+1] = ys[i]   # mantener último valor si hay error numérico
        xs[i+1] = xs[i] + h

    return xs, ys


# ============================================================
#  SOLUCIÓN NUMÉRICA + GRÁFICA
# ============================================================

def numeric_solve_and_plot(eq, initials_text=None, x_span=(0, 10)):
    x = symbols('x')
    y = Function('y')

    expr = sp.simplify(eq.lhs - eq.rhs)

    # Detectar orden
    derivatives = list(expr.atoms(sp.Derivative))
    if not derivatives:
        raise ValueError(
            "No se detectaron derivadas en la ecuación.\n"
            "Ejemplos válidos: y' = x + y   |   y'' + 4y = 0   |   y' - 2y = e^x"
        )
    max_order = max(d.derivative_count for d in derivatives)

    # Despejar derivada de mayor orden
    highest = [d for d in derivatives if d.derivative_count == max_order][0]
    sol_highest = solve(sp.Eq(expr, 0), highest)
    if not sol_highest:
        raise ValueError("No se pudo despejar la derivada principal. Reformula la ecuación.")
    rhs_expr = sol_highest[0]

    # Variables auxiliares
    y_syms = [sp.Symbol(f'_y{i}') for i in range(max_order)]
    replacements = {sp.Derivative(y(x), (x, k)): y_syms[k] for k in range(max_order)}
    rhs_expr = rhs_expr.subs(replacements)

    f_rhs = sp.lambdify([x, *y_syms], rhs_expr, modules='numpy')

    def f_system(xi, Yi):
        dY = np.zeros(max_order)
        for k in range(max_order - 1):
            dY[k] = Yi[k + 1]
        dY[-1] = float(f_rhs(xi, *Yi))
        return dY

    # Condiciones iniciales
    parsed = parse_initial_conditions(initials_text)
    if parsed:
        x0, ics = parsed
        y0 = [ics.get(i, 0.0) for i in range(max_order)]
    else:
        x0 = x_span[0]
        y0 = [1.0] + [0.0] * (max_order - 1)

    xs, ys = euler_solve(f_system, x0, y0, x_span[1], n_steps=1000)

    # ── Gráfica ──
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor('#10101c')
    ax.set_facecolor('#10101c')

    colors = ['#f97316', '#38bdf8', '#a78bfa', '#34d399']
    labels = ['y(x)', "y'(x)", "y''(x)", "y'''(x)"]
    for i in range(min(ys.shape[1], 4)):
        ax.plot(xs, ys[:, i], color=colors[i], linewidth=2, label=labels[i])

    ax.set_xlabel('x', color='#94a3b8', fontsize=11)
    ax.set_ylabel('y(x)', color='#94a3b8', fontsize=11)
    ax.tick_params(colors='#475569', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1e293b')
    ax.grid(True, color='#1e293b', linestyle='--', linewidth=0.6, alpha=0.8)
    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0', fontsize=9)

    plt.tight_layout()
    os.makedirs('static/plots', exist_ok=True)
    filename = f"plot_{int(time.time()*1000)}.png"
    path = f"static/plots/{filename}"
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()

    return f"plots/{filename}", max_order


# ============================================================
#  ROUTE
# ============================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    solution       = None
    solution_latex = None
    error          = None
    plot_path      = None
    eq_display     = None
    numeric_info   = None

    if request.method == 'POST':
        try:
            eq_input = request.form.get('equation', '').strip()
            initials = request.form.get('initials', '').strip()

            x_start = float(request.form.get('x_start', '0') or '0')
            x_end   = float(request.form.get('x_end',  '10') or '10')
            if x_end <= x_start:
                x_end = x_start + 10

            eq = parse_human_input(eq_input)
            eq_display = sp.latex(eq)

            # Intentar solución simbólica primero
            try:
                sol = dsolve(eq)
                solution = str(sol)
                solution_latex = sp.latex(sol)
            except Exception:
                # Fallback: Método de Euler
                plot, order = numeric_solve_and_plot(eq, initials, (x_start, x_end))
                plot_path = url_for('static', filename=plot)
                solution  = "Solución numérica por Método de Euler"
                numeric_info = {
                    'order': order,
                    'x_start': x_start,
                    'x_end': x_end,
                    'steps': 1000,
                }

        except Exception as e:
            error = str(e)

    return render_template(
        'index.html',
        solution=solution,
        solution_latex=solution_latex,
        error=error,
        plot_path=plot_path,
        eq_display=eq_display,
        numeric_info=numeric_info,
    )


if __name__ == '__main__':
    app.run(debug=True)