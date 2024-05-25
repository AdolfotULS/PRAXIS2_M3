import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import quad


# UTILIDADES


def curva_r(t):
    return np.array([t * np.cos(t), t * np.sin(t), t])


def funcion_interseccion(t):
    r = curva_r(t)
    return r[0] ** 2 + r[1] ** 2 + r[2] ** 2 - 2


# PARTE A Demostrar que la curva r(t) = (t cos(t), t sin(t), t) se encuentra sobre un cono, y visualizar esta curva y el cono usando herramientas de graficos 3D.


def visualizar_curva_y_cono():
    # Genera los datos de la curva r(t) para graficar
    valores_t = np.linspace(-10, 10, 400)
    valores_x, valores_y, valores_z = (
        valores_t * np.cos(valores_t),
        valores_t * np.sin(valores_t),
        valores_t,
    )

    # Crea la figura y los ejes para la grafica 3D
    figura = plt.figure()
    ax = figura.add_subplot(111, projection='3d')

    # Grafica la curva r(t)
    ax.plot(valores_x, valores_y, valores_z, label='Curva r(t)')

    # Genera los datos del cono para graficar
    radio_base = 1  # Radio de la base del cono
    altura = 5  # Altura del cono
    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, 5, 100)  # Rango de t es de 0 a 5
    U, V = np.meshgrid(u, v)
    X, Y, Z = (V * np.cos(U), V * np.sin(U), V)  # Sustituyendo las componentes de r(t)

    # Grafica la superficie del cono
    ax.plot_surface(X, Y, Z, alpha=0.5, color='pink')
    ax.set_title('Curva r(t) y Cono')
    ax.legend()

    # Solicitar al usuario los valores de t para la comprobacion
    print("Ingrese 5 valores para t , separados por comas:")
    input_t_values = input()
    t_values = [float(t) for t in input_t_values.split(",") if -100 <= float(t) <= 100][:5]
    
    puntos_x, puntos_y, puntos_z = [], [], []  #Almacenar valores

    for t in t_values:
        x = t * np.cos(t)
        y = t * np.sin(t)
        z = t
        puntos_x.append(x)
        puntos_y.append(y)
        puntos_z.append(z)
        comprobacion = x**2 + y**2 - z**2
        print(f"\nPara t = {t}, x^2 + y^2 - z^2 = {comprobacion}")

        if np.isclose(comprobacion, 0):
            print(f"La curva r(t) se encuentra sobre la superficie del cono para t = {t}.\n")
        else:
            print(f"La curva r(t) NO se encuentra sobre la superficie del cono para t = {t}.\n")

    # Marca los puntos de comprobacion en la grafica
    ax.scatter(puntos_x, puntos_y, puntos_z, color='red', s=50, label='Puntos de Comprobacion')
    ax.legend()

    plt.show()


# PARTE B


def interseccion_con_esfera():
    # Lista para almacenar los valores de t en los puntos de interseccion
    t_intersecciones = []
    # Itera sobre una serie de valores iniciales para encontrar soluciones
    for estimacion in np.linspace(-6, 6, 20):
        try:
            # Busca una solucion usando el valor inicial
            t_sol = fsolve(funcion_interseccion, estimacion)
            # Verifica si la solucion es valida y no esta duplicada
            if np.isclose(funcion_interseccion(t_sol), 0) and not any(
                np.isclose(t_sol, t, atol=1e-5) for t in t_intersecciones
            ):
                t_intersecciones.append(t_sol[0])
        except:
            continue

    # Calcula los puntos de interseccion en el espacio
    puntos_interseccion = np.array([curva_r(t) for t in t_intersecciones])

    # Imprime los resultados de las intersecciones encontradas
    print("Valores calculados para interseccion con la esfera:")
    for i, (t, punto) in enumerate(zip(t_intersecciones, puntos_interseccion)):
        print(
            f"Interseccion {i + 1}: t = {t:.5f}, (x, y, z) = ({punto[0]:.5f}, {punto[1]:.5f}, {punto[2]:.5f})"
        )

    # Genera los datos de la curva r(t) para graficar
    valores_t = np.linspace(-5, 5, 400)
    valores_x, valores_y, valores_z = (
        valores_t * np.cos(valores_t),
        valores_t * np.sin(valores_t),
        valores_t,
    )

    # Crea la figura y los ejes para la grafica 3D
    figura = plt.figure()
    ax = figura.add_subplot(111, projection="3d")
    # Grafica la curva r(t)
    ax.plot(valores_x, valores_y, valores_z, label="Curva r(t)")
    # Grafica los puntos de interseccion si existen
    if puntos_interseccion.size > 0:
        ax.scatter(
            puntos_interseccion[:, 0],
            puntos_interseccion[:, 1],
            puntos_interseccion[:, 2],
            color="red",
            label="Interseccion",
        )

    # Genera la superficie de la esfera para graficar
    u, v = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
    U, V = np.meshgrid(u, v)
    X, Y, Z = (
        np.sqrt(2) * np.sin(V) * np.cos(U),
        np.sqrt(2) * np.sin(V) * np.sin(U),
        np.sqrt(2) * np.cos(V),
    )

    # Grafica la superficie de la esfera
    ax.plot_surface(X, Y, Z, alpha=0.5, color="blue")
    ax.set_title("Interseccion con Esfera")
    ax.legend()
    plt.show()


# PARTE C
def longitud_de_la_curva():
    # Define la derivada de la curva r(t)
    def derivada_curva_r(t):
        return np.array([np.cos(t) - t * np.sin(t), np.sin(t) + t * np.cos(t), 1])

    # Calcula la norma de un vector
    def norma_vector(v):
        return np.sqrt(np.sum(v**2))

    # Funcion a integrar para calcular la longitud de la curva
    def integrando(t):
        return norma_vector(derivada_curva_r(t))

    # Encontrar los puntos de interseccion con la esfera
    t_intersecciones = []
    for estimacion in np.linspace(0, 10, 20):
        try:
            t_sol = fsolve(funcion_interseccion, estimacion)
            if np.isclose(funcion_interseccion(t_sol), 0) and not any(
                np.isclose(t_sol, t, atol=1e-5) for t in t_intersecciones
            ):
                t_intersecciones.append(t_sol[0])
        except:
            continue

    if not t_intersecciones:
        print("No se encontraron puntos de interseccion.")
        return

    puntos_interseccion = np.array([curva_r(t) for t in t_intersecciones])

    # Mostrar los resultados de las intersecciones
    print("Valores calculados para la longitud de la curva:")
    for i, (t, punto) in enumerate(zip(t_intersecciones, puntos_interseccion)):
        longitud, _ = quad(integrando, 0, t)
        print(
            f"Interseccion {i + 1}: t = {t:.5f}, (x, y, z) = ({punto[0]:.5f}, {punto[1]:.5f}, {punto[2]:.5f}), Longitud: {longitud:.5f}"
        )

    # Generar datos para graficar la curva r(t) hasta el primer punto de interseccion
    t_vals = np.linspace(0, t_intersecciones[0], 400)
    x_vals, y_vals, z_vals = t_vals * np.cos(t_vals), t_vals * np.sin(t_vals), t_vals

    # Crear figura y ejes para la grafica 3D
    figura = plt.figure()
    ax = figura.add_subplot(111, projection="3d")
    # Graficar la curva r(t) hasta el primer punto de interseccion
    ax.plot(x_vals, y_vals, z_vals, label="Longitud de Curva", color="green")

    # Generar datos para graficar la curva r(t) despues del primer punto de interseccion
    t_vals = np.linspace(t_intersecciones[0], 4, 400)
    x_vals, y_vals, z_vals = t_vals * np.cos(t_vals), t_vals * np.sin(t_vals), t_vals

    # Graficar la curva r(t)
    ax.plot(x_vals, y_vals, z_vals, label="Curva r(t)", color="blue")

    # Graficar los puntos de interseccion
    if puntos_interseccion.size > 0:
        ax.scatter(
            puntos_interseccion[:, 0],
            puntos_interseccion[:, 1],
            puntos_interseccion[:, 2],
            color="red",
            label="Interseccion",
        )

    ax.set_title("Interseccion con Esfera y Longitud de la Curva")
    ax.legend()
    plt.show()


# MENU


def main():
    print("Praxis 2 | Adolfo Toledo - Ignacia Miranda")
    print("[1] Verificación Geométrica y Visualización, Curva r(t) y Cono")
    print("[2] Intersección con una Esfera")
    print("[3] Longitud de la Curva")
    opcion = input("Opción: ")

    while opcion not in ["1", "2", "3"]:
        print("[!] Opción Inválida")
        opcion = input("Opción: ")

    if opcion == "1":
        visualizar_curva_y_cono()
    elif opcion == "2":
        interseccion_con_esfera()
    elif opcion == "3":
        longitud_de_la_curva()


if __name__ == "__main__":
    main()
