from flask import Flask, render_template, request, url_for
from sympy import symbols, Function, Eq, dsolve, Derivative, sin, cos, exp, tan, log, solve
from sympy.parsing.sympy_parser import parse_expr
import sympy as sp
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
import time

app = Flask(__name__)

# ================= PARSER =================
def parse_human_input(user_input):
    x = symbols('x')
    y = Function('y')

    s = user_input.strip()
    s = s.replace('’', "'")
    s = s.replace('\u2212', '-')
    s = s.replace("^", "**")

    # Derivadas
    s = re.sub(r"y\s*''", "Derivative(y(x), x, x)", s)
    s = re.sub(r"y\s*'", "Derivative(y(x), x)", s)

    # Multiplicaciones implícitas (hacer antes de convertir 'y' a 'y(x)')
    s = re.sub(r"(?P<num>\d+(?:\.\d+)?)\s*(?=[A-Za-z\(])", r"\g<num>*", s)
    s = re.sub(r"\)\s*(?=[A-Za-z0-9\(])", r")*", s)
    # Solo insertar '*' cuando el carácter anterior sea un dígito o ')' para no romper llamadas a funciones
    s = re.sub(r"(?<=[0-9\)])\s*(?=[A-Za-z\(])", r"*", s)

    # y → y(x) (después de las reglas de multiplicación)
    s = re.sub(r"\by\b(?!\s*\()", "y(x)", s)

    # Separar ecuación
    if '=' in s:
        lhs_s, rhs_s = s.split('=', 1)
    else:
        lhs_s, rhs_s = s, '0'

    # 🔥 CORREGIR: mapear 'y' a la función y (Function('y')) en local_dict
    local_dict = {
        'x': x,
        'y': y,
        'sin': sin,
        'cos': cos,
        'exp': exp,
        'tan': tan,
        'log': log,
        'Derivative': Derivative,
    }

    lhs = parse_expr(lhs_s, local_dict=local_dict)
    rhs = parse_expr(rhs_s, local_dict=local_dict)

    return Eq(lhs, rhs)


# ================= CONDICIONES =================
def parse_initial_conditions(initials_text):
    if not initials_text:
        return None

    ic_regex = re.compile(r"y(?P<primes>'*)\((?P<x>[^)]+)\)\s*=\s*(?P<val>[-+]?\d*\.?\d+)")
    matches = ic_regex.finditer(initials_text)

    ics = {}
    x0 = None

    for m in matches:
        order = len(m.group('primes'))
        x_val = float(m.group('x'))
        val = float(m.group('val'))
        ics[order] = val
        x0 = x_val

    return (x0, ics) if ics else None


# ================= NUMÉRICO =================
def numeric_solve_and_plot(eq, initials_text=None, x_span=(0, 10)):
    x = symbols('x')
    y = Function('y')

    expr = sp.simplify(eq.lhs - eq.rhs)

    derivatives = list(expr.atoms(sp.Derivative))
    if not derivatives:
        raise ValueError("No se detectaron derivadas; escribe la ecuación usando y' o y'' o y(x) y sus derivadas.")
    max_order = max([d.derivative_count for d in derivatives])

    highest = [d for d in derivatives if d.derivative_count == max_order][0]

    sol = solve(sp.Eq(expr, 0), highest)
    rhs = sol[0]

    y_symbols = [sp.Symbol(f'y{i}') for i in range(max_order)]

    replacements = {
        sp.Derivative(y(x), (x, k)): y_symbols[k]
        for k in range(max_order)
    }

    rhs = rhs.subs(replacements)
    func = sp.lambdify((x, *y_symbols), rhs, 'numpy')

    parsed = parse_initial_conditions(initials_text)

    if parsed:
        x0, ics = parsed
        y0 = [ics.get(i, 0) for i in range(max_order)]
    else:
        x0 = x_span[0]
        y0 = [1] + [0]*(max_order-1)

    def system(t, Y):
        dY = np.zeros_like(Y)
        for i in range(max_order - 1):
            dY[i] = Y[i+1]
        dY[-1] = func(t, *Y)
        return dY

    t = np.linspace(x_span[0], x_span[1], 400)
    sol = solve_ivp(system, x_span, y0, t_eval=t)

    # Gráfica
    plt.figure()
    plt.plot(sol.t, sol.y[0])
    plt.grid()

    os.makedirs('static/plots', exist_ok=True)
    filename = f"plot_{int(time.time())}.png"
    path = f"static/plots/{filename}"
    plt.savefig(path)
    plt.close()

    return f"plots/{filename}"


# ================= ROUTE =================
@app.route('/', methods=['GET', 'POST'])
def index():
    solution = None
    solution_latex = None
    error = None
    plot_path = None

    if request.method == 'POST':
        try:
            eq_input = request.form['equation']
            initials = request.form.get('initials', '')

            # Obtener rangos x de forma segura (evitar float('') error)
            x_start_raw = request.form.get('x_start', '').strip()
            x_end_raw = request.form.get('x_end', '').strip()
            try:
                x_start = float(x_start_raw) if x_start_raw != '' else 0.0
            except ValueError:
                x_start = 0.0
            try:
                x_end = float(x_end_raw) if x_end_raw != '' else 10.0
            except ValueError:
                x_end = 10.0

            eq = parse_human_input(eq_input)

            try:
                sol = dsolve(eq)
                solution = str(sol)
                try:
                    solution_latex = sp.latex(sol)
                except Exception:
                    solution_latex = None
            except Exception:
                plot = numeric_solve_and_plot(eq, initials, (x_start, x_end))
                plot_path = url_for('static', filename=plot)
                solution = "Solución numérica mostrada abajo"

        except Exception as e:
            error = str(e)

    return render_template('index.html', solution=solution, solution_latex=solution_latex, error=error, plot_path=plot_path)


if __name__ == '__main__':
    app.run(debug=True)