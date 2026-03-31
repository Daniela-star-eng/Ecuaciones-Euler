from flask import Flask, render_template, request, url_for
import sympy as sp
from sympy import (symbols, Function, Eq, dsolve, Derivative,
                   sin, cos, exp, tan, log, solve, sqrt, pi, E, latex)
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

# ============================================================
#  CONSTANTES
# ============================================================

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Todos los caracteres Unicode que pueden parecer un apóstrofe/prima
_APOSTROPHES = (
    '\u2018', '\u2019',  # comillas simples curvas izq/der
    '\u02bc', '\u02b9',  # modifier letter apostrophe / prime
    '\u02be', '\u02bf',  # modifier letter right/left half ring
    '\u0060', '\u00b4',  # grave accent, acute accent
    '\u2032', '\u2033',  # prime, double prime
    '\uff07',            # fullwidth apostrophe
    '\u275c',            # heavy single comma quotation mark
)


def _normalize_apostrophes(s: str) -> str:
    """Reemplaza TODOS los apóstrofes/primas Unicode por el ASCII estándar '."""
    for ch in _APOSTROPHES:
        s = s.replace(ch, "'")
    return s


# ============================================================
#  PARSER DE ECUACIÓN
# ============================================================

def normalize_input(s: str) -> str:
    s = _normalize_apostrophes(s)
    s = s.replace('\u2212', '-').replace('\u00b2', '**2').replace('\u00b3', '**3')
    s = s.replace('^', '**')
    s = s.replace('/', ' / ')
    s = s.strip()
    return re.sub(r'\s+', ' ', s)


def replace_functions(s: str) -> str:
    aliases = {
        r'\bsen\b': 'sin', r'\bSin\b': 'sin',
        r'\bCos\b': 'cos', r'\bTan\b': 'tan',
        r'\bExp\b': 'exp', r'\bln\b':  'log',
    }
    for pat, repl in aliases.items():
        s = re.sub(pat, repl, s)
    return s


def replace_derivatives(s: str) -> str:
    s = re.sub(r"dy\s*/\s*dx", "Derivative(y(x), x)", s)
    s = re.sub(r"d\^?(\d+)\s*y\s*/\s*d[xt]\^?\1",
               lambda m: f"Derivative(y(x), x, {m.group(1)})", s)
    s = re.sub(r"y\s*'''", "Derivative(y(x), x, 3)", s)
    s = re.sub(r"y\s*''",  "Derivative(y(x), x, 2)", s)
    s = re.sub(r"y\s*'",   "Derivative(y(x), x)",    s)
    return s


def fix_implicit_multiplication(s: str) -> str:
    s = re.sub(r'(\d+\.?\d*)\s*([a-zA-Z\(])', r'\1*\2', s)
    s = re.sub(r'\)\s*([a-zA-Z0-9\(])',        r')*\1',  s)
    return s


def replace_standalone_y(s: str) -> str:
    return re.sub(r'\by\b(?!\s*[\(\'\'])', 'y(x)', s)


def parse_human_input(user_input: str) -> Eq:
    x = symbols('x')
    y = Function('y')

    local_dict = {
        'x': x, 'y': y,
        'sin': sin, 'cos': cos, 'exp': exp,
        'tan': tan, 'log': log, 'sqrt': sqrt,
        'pi': pi, 'e': E,
        'Derivative': Derivative,
    }

    s = normalize_input(user_input)
    s = replace_functions(s)
    s = replace_derivatives(s)
    s = fix_implicit_multiplication(s)
    s = replace_standalone_y(s)

    lhs_s, rhs_s = s.split('=', 1) if '=' in s else (s, '0')

    lhs = parse_expr(lhs_s, local_dict=local_dict, transformations=TRANSFORMATIONS)
    rhs = parse_expr(rhs_s, local_dict=local_dict, transformations=TRANSFORMATIONS)
    return Eq(lhs, rhs)


# ============================================================
#  PARSE DE CONDICIONES INICIALES
# ============================================================

def parse_initial_conditions(initials_str: str) -> dict:
    """
    Acepta: 'y(0)=693.36'  o  'y(0)=1; y\'(0)=0'
    Devuelve: { (x0: float, orden: int): valor: float }
    """
    if not initials_str or not initials_str.strip():
        return {}

    # Normalizar apóstrofes Unicode también en el campo de CIs
    initials_str = _normalize_apostrophes(initials_str)

    conditions = {}
    for part in re.split(r'[;,]', initials_str):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"y(\s*'*)?\s*\(\s*([^\)]+)\s*\)\s*=\s*(.+)", part)
        if m:
            primes  = m.group(1) or ''
            x0_str  = m.group(2).strip()
            val_str = m.group(3).strip()
            order   = primes.count("'")
            try:
                conditions[(float(x0_str), order)] = float(val_str)
            except ValueError:
                pass

    return conditions


# ============================================================
#  APLICAR CIs SIMBÓLICAMENTE
# ============================================================

def apply_initial_conditions_symbolic(general_sol, conditions: dict, x_sym):
    if not conditions:
        return general_sol

    constants = sorted(general_sol.rhs.free_symbols - {x_sym}, key=str)
    if not constants:
        return general_sol

    y = Function('y')
    eqs = []
    for (x0, order), y_val in conditions.items():
        if order == 0:
            expr = general_sol.rhs.subs(x_sym, x0) - y_val
        else:
            deriv = sp.diff(general_sol.rhs, x_sym, order)
            expr  = deriv.subs(x_sym, x0) - y_val
        eqs.append(expr)

    try:
        solution = solve(eqs, constants)
        if isinstance(solution, list) and solution:
            s0 = solution[0]
            solution = s0 if isinstance(s0, dict) else dict(
                zip(constants, s0 if hasattr(s0, '__iter__') else [s0])
            )
        elif not isinstance(solution, dict):
            solution = {}

        particular_rhs = sp.simplify(general_sol.rhs.subs(solution))
        return Eq(y(x_sym), particular_rhs)
    except Exception:
        return general_sol


# ============================================================
#  VECTOR y0 PARA EULER
# ============================================================

def get_y0_from_conditions(conditions: dict, x_start: float, max_order: int) -> list:
    """
    Construye el vector inicial para Euler a partir de las condiciones iniciales.
    Si no hay ninguna condición en x_start, usa y(x_start)=1 como fallback.
    """
    y0 = [0.0] * max_order
    applied_any = False

    for (x0, order), val in conditions.items():
        if abs(x0 - x_start) < 1e-10 and order < max_order:
            y0[order] = val
            applied_any = True

    if not applied_any:
        y0[0] = 1.0  # fallback solo si el usuario no dio ninguna CI

    return y0


# ============================================================
#  EULER
# ============================================================

def euler_solve(f_system, x0: float, y0: list, x_end: float, n_steps: int = 1000):
    h  = (x_end - x0) / n_steps
    n  = len(y0)
    xs = np.zeros(n_steps + 1)
    ys = np.zeros((n_steps + 1, n))

    xs[0] = x0
    ys[0] = y0

    for i in range(n_steps):
        dY        = f_system(xs[i], ys[i])
        ys[i + 1] = ys[i] + h * np.array(dY)
        xs[i + 1] = xs[i] + h

    return xs, ys


# ============================================================
#  SOLUCIÓN NUMÉRICA + GRÁFICA
# ============================================================

def numeric_solve_and_plot(eq: Eq, x_span=(0, 10), y0_override=None):
    x = symbols('x')
    y = Function('y')

    # Obtener derivadas — probar primero con simplify, luego sin él
    expr = sp.simplify(eq.lhs - eq.rhs)
    derivatives = list(expr.atoms(sp.Derivative))
    if not derivatives:
        expr = eq.lhs - eq.rhs
        derivatives = list(expr.atoms(sp.Derivative))
    if not derivatives:
        raise ValueError(
            "No se detectaron derivadas en la ecuación. "
            "Asegúrate de escribir y', y'' o dy/dx."
        )

    max_order = max(d.derivative_count for d in derivatives)
    highest   = next(d for d in derivatives if d.derivative_count == max_order)
    rhs_expr  = solve(sp.Eq(expr, 0), highest)[0]

    # Sustituir Derivative(y(x),(x,k)) → símbolo escalar _yk
    y_syms       = [sp.Symbol(f'_y{i}') for i in range(max_order)]
    replacements = {sp.Derivative(y(x), (x, k)): y_syms[k] for k in range(max_order)}
    # y(x) mismo → _y0
    replacements[y(x)] = y_syms[0]
    rhs_expr = rhs_expr.subs(replacements)

    f_rhs = sp.lambdify([x, *y_syms], rhs_expr, modules='numpy')

    def f_system(xi, Yi):
        dY = np.zeros(max_order)
        for k in range(max_order - 1):
            dY[k] = Yi[k + 1]
        dY[-1] = float(f_rhs(xi, *Yi))
        return dY

    # ── Construir y0 ──────────────────────────────────────────
    if y0_override is not None:
        y0 = list(y0_override)
        while len(y0) < max_order:
            y0.append(0.0)
        y0 = y0[:max_order]
    else:
        y0 = [1.0] + [0.0] * (max_order - 1)

    x0 = x_span[0]
    xs, ys = euler_solve(f_system, x0, y0, x_span[1])

    # ── Gráfica ───────────────────────────────────────────────
    plt.figure(facecolor='#0f172a')
    ax = plt.gca()
    ax.set_facecolor('#0f172a')
    ax.plot(xs, ys[:, 0], color='#60a5fa', linewidth=2)
    ax.grid(color='#1e293b', linewidth=0.8)
    ax.tick_params(colors='#9ca3af')
    ax.spines[:].set_color('#1e293b')
    ax.set_xlabel('x', color='#9ca3af')
    ax.set_ylabel('y', color='#9ca3af')
    ax.set_title('Solución numérica — Método de Euler', color='#e5e7eb', fontsize=11)

    os.makedirs('static/plots', exist_ok=True)
    filename = f"plot_{int(time.time() * 1000)}.png"
    path     = f"static/plots/{filename}"
    plt.savefig(path, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

    return f"plots/{filename}", xs, ys


# ============================================================
#  RUTA PRINCIPAL
# ============================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    solution        = None
    solution_latex  = None
    particular_latex = None
    plot_path       = None
    table_data      = None
    point_result    = None
    solution_method = None
    initials_parsed = {}

    if request.method == 'POST':
        try:
            eq_input     = request.form.get('equation', '')
            x_eval       = request.form.get('x_eval', '')
            initials_str = request.form.get('initials', '')
            x_start      = float(request.form.get('x_start') or 0)
            x_end        = float(request.form.get('x_end')   or 10)

            eq = parse_human_input(eq_input)

            # Condiciones iniciales
            initials_parsed = parse_initial_conditions(initials_str)

            # ── Orden de la ODE (para construir y0 antes de llamar Euler) ──
            try:
                _expr  = eq.lhs - eq.rhs          # sin simplify para no perder Derivatives
                _derivs = list(_expr.atoms(sp.Derivative))
                if not _derivs:
                    _expr  = sp.simplify(_expr)
                    _derivs = list(_expr.atoms(sp.Derivative))
                ode_order = max(d.derivative_count for d in _derivs) if _derivs else 1
            except Exception:
                ode_order = 1

            # y0 para Euler — usa las CIs del formulario
            y0_for_euler = get_y0_from_conditions(initials_parsed, x_start, ode_order)

            # ── Solución simbólica ──────────────────────────────────────────
            general_sol  = None
            particular_sol = None

            try:
                x_sym       = symbols('x')
                general_sol = dsolve(eq)
                solution_latex  = latex(general_sol)
                solution_method = "Solución Simbólica"

                if initials_parsed:
                    particular_sol  = apply_initial_conditions_symbolic(
                        general_sol, initials_parsed, x_sym
                    )
                    particular_latex = latex(particular_sol)

            except Exception:
                solution_method = "Solución Numérica (Método de Euler)"

            # ── Euler ──────────────────────────────────────────────────────
            plot, xs, ys = numeric_solve_and_plot(
                eq, (x_start, x_end), y0_override=y0_for_euler
            )
            plot_path = url_for('static', filename=plot)

            # Tabla (máx 100 filas)
            table_data = []
            step = max(1, len(xs) // 100)
            for i in range(0, len(xs), step):
                table_data.append({
                    'i': i,
                    'x': round(float(xs[i]),    4),
                    'y': round(float(ys[i][0]), 4),
                })

            # Punto evaluado
            if x_eval and x_eval.strip():
                try:
                    x_val = float(x_eval)

                    y_particular_val = None
                    if particular_sol is not None:
                        try:
                            y_particular_val = float(
                                particular_sol.rhs.subs(symbols('x'), x_val)
                            )
                        except Exception:
                            pass

                    if x_start <= x_val <= x_end:
                        y_euler = float(np.interp(x_val, xs, ys[:, 0]))
                        point_result = {
                            'x':          x_val,
                            'y':          round(y_euler, 6),
                            'y_symbolic': round(y_particular_val, 6) if y_particular_val is not None else None,
                            'method':     solution_method,
                        }
                    else:
                        point_result = {
                            'x': x_val, 'y': None, 'y_symbolic': None,
                            'error': f'x debe estar entre {x_start} y {x_end}',
                        }
                except ValueError as e:
                    point_result = {
                        'x': x_eval, 'y': None, 'y_symbolic': None,
                        'error': f'Valor inválido: {e}',
                    }

        except Exception as e:
            solution        = f"Error al procesar: {e}"
            solution_method = "Error"

    return render_template(
        'index.html',
        solution         = solution,
        solution_latex   = solution_latex,
        particular_latex = particular_latex,
        plot_path        = plot_path,
        table_data       = table_data,
        point_result     = point_result,
        solution_method  = solution_method,
        initials_parsed  = initials_parsed,
    )


if __name__ == '__main__':
    app.run(debug=True)