from flask import Flask, render_template, request, url_for
import sympy as sp
from sympy import symbols, Function, Eq, dsolve, Derivative, sin, cos, exp, tan, log, solve, sqrt, pi, E, latex
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
#  PARSER
# ============================================================

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

def normalize_input(s: str) -> str:
    s = s.replace('\u2018', "'").replace('\u2019', "'")
    s = s.replace('\u2212', '-').replace('\u00b2', '**2').replace('\u00b3', '**3')
    s = s.replace('^', '**')
    # Agregar soporte para fracciones como dy/dx
    s = s.replace('/', ' / ')
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def replace_derivatives(s: str) -> str:
    # Manejo de dy/dx = ... (ecuaciones de primer orden)
    s = re.sub(r"dy\s*/\s*dx", "Derivative(y(x), x)", s)
    s = re.sub(r"d\^?(\d+)\s*y\s*/\s*d[xt]\^?\1", lambda m: f"Derivative(y(x), x, {m.group(1)})", s)
    s = re.sub(r"y\s*'''", "Derivative(y(x), x, 3)", s)
    s = re.sub(r"y\s*''", "Derivative(y(x), x, 2)", s)
    s = re.sub(r"y\s*'", "Derivative(y(x), x)", s)
    return s

def replace_functions(s: str) -> str:
    aliases = {
        r'\bsen\b': 'sin', r'\bSin\b': 'sin',
        r'\bCos\b': 'cos', r'\bTan\b': 'tan',
        r'\bExp\b': 'exp', r'\bln\b': 'log',
    }
    for pat, repl in aliases.items():
        s = re.sub(pat, repl, s)
    return s

def fix_implicit_multiplication(s: str) -> str:
    s = re.sub(r'(\d+\.?\d*)\s*([a-zA-Z\(])', r'\1*\2', s)
    s = re.sub(r'\)\s*([a-zA-Z0-9\(])', r')*\1', s)
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

    if '=' in s:
        lhs_s, rhs_s = s.split('=', 1)
    else:
        lhs_s, rhs_s = s, '0'

    lhs = parse_expr(lhs_s, local_dict=local_dict, transformations=TRANSFORMATIONS)
    rhs = parse_expr(rhs_s, local_dict=local_dict, transformations=TRANSFORMATIONS)
    return Eq(lhs, rhs)

# ============================================================
#  EULER
# ============================================================

def euler_solve(f_system, x0, y0, x_end, n_steps=1000):
    h  = (x_end - x0) / n_steps
    n  = len(y0)
    xs = np.zeros(n_steps + 1)
    ys = np.zeros((n_steps + 1, n))

    xs[0] = x0
    ys[0] = y0

    for i in range(n_steps):
        dY = f_system(xs[i], ys[i])
        ys[i+1] = ys[i] + h * np.array(dY)
        xs[i+1] = xs[i] + h

    return xs, ys

# ============================================================
#  NUMÉRICO + GRÁFICA
# ============================================================

def numeric_solve_and_plot(eq, x_span=(0, 10)):
    x = symbols('x')
    y = Function('y')

    expr = sp.simplify(eq.lhs - eq.rhs)
    derivatives = list(expr.atoms(sp.Derivative))
    max_order = max(d.derivative_count for d in derivatives)

    highest = [d for d in derivatives if d.derivative_count == max_order][0]
    rhs_expr = solve(sp.Eq(expr, 0), highest)[0]

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

    x0 = x_span[0]
    y0 = [1.0] + [0.0] * (max_order - 1)

    xs, ys = euler_solve(f_system, x0, y0, x_span[1])

    # gráfica
    plt.figure()
    plt.plot(xs, ys[:, 0])
    plt.grid()

    os.makedirs('static/plots', exist_ok=True)
    filename = f"plot_{int(time.time()*1000)}.png"
    path = f"static/plots/{filename}"
    plt.savefig(path)
    plt.close()

    return f"plots/{filename}", xs, ys

# ============================================================
#  ROUTE
# ============================================================

@app.route('/', methods=['GET', 'POST'])
def index():
    solution = None
    solution_latex = None
    plot_path = None
    table_data = None
    point_result = None
    solution_method = None

    if request.method == 'POST':
        try:
            eq_input = request.form.get('equation')
            x_eval = request.form.get('x_eval')

            x_start = float(request.form.get('x_start') or 0)
            x_end = float(request.form.get('x_end') or 10)

            eq = parse_human_input(eq_input)

            # simbólico
            try:
                sol = dsolve(eq)
                solution_latex = latex(sol)
                solution_method = "Solución Simbólica"
            except Exception as symbolic_error:
                solution_method = "Solución Numérica (Método de Euler)"
                solution = None

            # euler - siempre ejecutar para obtener valores numéricos
            plot, xs, ys = numeric_solve_and_plot(eq, (x_start, x_end))
            plot_path = url_for('static', filename=plot)

            # tabla
            table_data = []
            step = max(1, len(xs)//100)

            for i in range(0, len(xs), step):
                table_data.append({
                    'i': i,
                    'x': round(float(xs[i]), 4),
                    'y': round(float(ys[i][0]), 4)
                })

            # punto
            if x_eval and x_eval.strip():
                try:
                    x_val = float(x_eval)
                    if x_start <= x_val <= x_end:
                        y_interp = float(np.interp(x_val, xs, ys[:, 0]))
                        point_result = {
                            'x': x_val, 
                            'y': round(y_interp, 6),
                            'method': solution_method
                        }
                    else:
                        point_result = {
                            'x': x_val,
                            'y': None,
                            'error': f'x debe estar entre {x_start} y {x_end}'
                        }
                except ValueError as e:
                    point_result = {
                        'x': x_eval,
                        'y': None,
                        'error': f'Error: {str(e)}'
                    }

        except Exception as e:
            solution = f"Error al procesar: {str(e)}"
            solution_method = "Error"

    return render_template(
        'index.html',
        solution=solution,
        solution_latex=solution_latex,
        plot_path=plot_path,
        table_data=table_data,
        point_result=point_result,
        solution_method=solution_method
    )

if __name__ == '__main__':
    app.run(debug=True)