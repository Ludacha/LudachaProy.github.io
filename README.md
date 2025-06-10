Introducción
Resumen del Proyecto: Solución de Problemas de Ingeniería y Ciencia Mediante la Búsqueda de Raíces con Métodos Numéricos

Este proyecto se enfoca en la aplicación de métodos numéricos para resolver problemas específicos en ingeniería y ciencia, encontrando las raíces de las ecuaciones que los modelan. Resolveremos tres problemas concretos:

1. Encontrar el punto de operación de un diodo en un circuito eléctrico, utilizando la ecuación de Shockley.
2. Optimizar una función de costo en un proceso de producción, determinando el punto donde la derivada de la función de costo es cero.
3. Modelar el equilibrio químico de una reacción reversible, calculando las concentraciones de los reactivos y productos en el equilibrio.

Para abordar estos problemas, implementaremos y compararemos tres métodos numéricos: Bisección, Newton-Raphson y Secante. El proyecto incluye código Python para implementar los métodos, visualizaciones gráficas de las
funciones y las raíces, y un análisis comparativo del rendimiento de cada método. El código fuente está disponible en un repositorio Git, y se ha creado una página web interactiva para permitir a los usuarios experimentar 
con los métodos y parámetros. El objetivo es demostrar la utilidad práctica de estos métodos para resolver problemas reales en ingeniería y ciencia.

1. Diseño de Circuitos Eléctricos (Diodo)
Problema
Un diodo tiene una corriente de saturación inversa Is = 1e-12 A y un voltaje térmico Vt = 0.026 V. Está conectado en serie con una resistencia de R = 100 Ohmios y una fuente de voltaje de V = 5 V. Determina el voltaje en
el diodo (Vd) utilizando el método de Newton-Raphson.
🧮 Código en Python

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del diodo
Is = 1e-12  # Corriente de saturación inversa (A)
Vt = 0.026  # Voltaje térmico (V)
R = 100     # Resistencia (Ohmios)
V = 5       # Voltaje de la fuente (V)

# Ecuación del diodo (con resistencia en serie)
def diode_equation(Vd):
    return Is * (np.exp(Vd / Vt) - 1) + (Vd / R) - (V / R)

# Derivada de la ecuación del diodo
def diode_equation_derivative(Vd):
    return (Is / Vt) * np.exp(Vd / Vt) + (1 / R)

# Método de Newton-Raphson
def newton_raphson(f, f_prime, x0, tol=1e-6, max_iter=100):
    x = x0
    iterations = [x0]
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x, i + 1, iterations
        f_prime_x = f_prime(x)
        if f_prime_x == 0:
            print("Derivada es cero. Newton-Raphson falló.")
            return None, i + 1, iterations
        x = x - fx / f_prime_x
        iterations.append(x)
    print("Newton-Raphson no convergió.")
    return None, max_iter, iterations

# Estimación inicial
Vd_initial = 0.7  # Voltios

# Resolver la ecuación
Vd, iterations, all_iterations = newton_raphson(diode_equation, diode_equation_derivative, Vd_initial)

if Vd is not None:
    print(f"Voltaje del diodo (Vd): {Vd:.6f} V")
    print(f"Número de iteraciones: {iterations}")

    # Graficar
    Vd_range = np.linspace(0, 0.8, 400)
    f_Vd = [diode_equation(v) for v in Vd_range]

    plt.figure(figsize=(8, 6))
    plt.plot(Vd_range, f_Vd, label="Ecuación del Diodo")
    plt.scatter(Vd, 0, color='red', label=f"Vd = {Vd:.3f} V")
    plt.xlabel("Voltaje del Diodo (Vd)")
    plt.ylabel("f(Vd)")
    plt.title("Punto de Operación del Diodo")
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("No se encontró la solución.")

  
Resultados.

🔹 Voltaje del diodo (vd): 0.636974 V

🔹 Nro de iteraciones: 7
![img1](https://github.com/user-attachments/assets/c021d02c-56e2-4c96-9edc-c4a3d3757f0e)


2. Optimización en Ingeniería (Costo Mínimo)
Problema

Una empresa tiene una función de costo total dada por C(x) = x^4 - 6x^2 + 8x + 10, donde x es la cantidad de unidades producidas. Determina la cantidad de unidades que minimiza el costo total utilizando el método de la Secante.
🧮 Código en Python

import numpy as np
import matplotlib.pyplot as plt

# Función de costo
def cost_function(x):
    return x**4 - 6*x**2 + 8*x + 10

# Derivada de la función de costo
def cost_function_derivative(x):
    return 4*x**3 - 12*x + 8

# Método de la Secante
def secant(f, x0, x1, tol=1e-6, max_iter=100):
    iterations = [x0, x1]
    for i in range(max_iter):
        fx0 = f(x0)
        fx1 = f(x1)
        if abs(fx1) < tol:
            return x1, i + 1, iterations
        if fx1 == fx0:
            print("Secante falló (división por cero).")
            return None, i + 1, iterations
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        iterations.append(x2)
        x0 = x1
        x1 = x2
    print("Secante no convergió.")
    return None, max_iter, iterations

# Estimaciones iniciales
x0 = 0.0
x1 = 2.0

# Resolver la ecuación
x_min, iterations, all_iterations = secant(cost_function_derivative, x0, x1)

if x_min is not None:
    print(f"Cantidad de unidades que minimiza el costo: {x_min:.6f}")
    print(f"Número de iteraciones: {iterations}")

    # Graficar
    x_range = np.linspace(-3, 3, 400)
    C_x = [cost_function(x) for x in x_range]

    plt.figure(figsize=(8, 6))
    plt.plot(x_range, C_x, label="Función de Costo C(x)")
    plt.scatter(x_min, cost_function(x_min), color='red', label=f"Mínimo en x = {x_min:.3f}")
    plt.xlabel("Cantidad de Unidades (x)")
    plt.ylabel("Costo Total C(x)")
    plt.title("Optimización de Costo")
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("No se encontró la solución.")

  
Resultados.

🔹 Cantidad de unidades que minimiza el costo: -2.000000

🔹 Nro de iteraciones: 2
![img2](https://github.com/user-attachments/assets/aaa35262-2152-4ba4-b3c5-7bb57e3b4de9)

3. Modelado de Reacciones Químicas (Equilibrio)
Problema

Considera la reacción reversible A + B <=> C. Inicialmente, las concentraciones de A y B son A0 = 1.0 M y B0 = 1.0 M, respectivamente. La constante de equilibrio es K = 2.0. Determina la concentración de C en el equilibrio (x) utilizando el método de Bisección.
🧮 Código en Python

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la reacción
K = 2.0   # Constante de equilibrio
A0 = 1.0  # Concentración inicial de A (M)
B0 = 1.0  # Concentración inicial de B (M)

# Ecuación de equilibrio
def equilibrium_equation(x):
    return (x**2) / ((A0 - x) * (B0 - x)) - K

# Método de Bisección
def bisection(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) >= 0:
        print("La función no cambia de signo en el intervalo.")
        return None, None, []

    iterations = []
    for i in range(max_iter):
        c = (a + b) / 2
        iterations.append(c)
        if f(c) == 0 or (b - a) / 2 < tol:
            return c, i + 1, iterations

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

    print("Bisección no convergió.")
    return None, max_iter, iterations

# Intervalo inicial
a = 0.0
b = 0.9  # La concentración de C no puede ser mayor que la de A o B

# Resolver la ecuación
x, iterations, all_iterations = bisection(equilibrium_equation, a, b)

if x is not None:
    print(f"Concentración de C en el equilibrio: {x:.6f} M")
    print(f"Número de iteraciones: {iterations}")

    # Graficar
    x_range = np.linspace(0, 0.9, 400)
    f_x = [equilibrium_equation(xi) for xi in x_range]

    plt.figure(figsize=(8, 6))
    plt.plot(x_range, f_x, label="Ecuación de Equilibrio")
    plt.scatter(x, 0, color='red', label=f"Concentración de C = {x:.3f} M")
    plt.xlabel("Concentración de C (x)")
    plt.ylabel("f(x)")
    plt.title("Equilibrio Químico")
    plt.grid(True)
    plt.legend()
    plt.show()
else:
    print("No se encontró la solución.")

  
Resultados.

🔹 Concentración de C en el equilibrio: 0.585787 M

🔹 Nro de iteraciones: 20
![img3](https://github.com/user-attachments/assets/55c6eba2-ee98-4ad4-b5e8-2fb7fd0a226d)




