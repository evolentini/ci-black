#!/usr/bin/env python3
# encoding: utf-8

import time
import numpy as np
from argparse import ArgumentParser


def LMS_Float(x_sig, d_sig, Mu, N_LMS: int = 256, error: int = 6):
    N = min([len(x_sig), len(d_sig)])

    w = np.zeros(N_LMS)
    x_fifo = np.zeros(N_LMS)

    # Senal de error
    e_sig = np.zeros(N)
    y_sig = np.zeros(N)
    c_sig = np.zeros(N)

    c_fifo = np.zeros(2)
    c_fifo_1 = np.zeros(N_LMS)
    c_fifo_2 = np.zeros(N_LMS)

    stabilized = 0
    for i in range(N):
        x_fifo[1:] = x_fifo[:-1]
        x_fifo[0] = x_sig[i]
        y_sig_acc = 0
        for j in range(N_LMS):
            y_sig_mul = x_fifo[j] * w[j]
            y_sig_acc += y_sig_mul
        y_sig[i] = y_sig_acc
        e_sig[i] = d_sig[i] - y_sig[i]
        for j in range(N_LMS - 1):
            if j > i:
                w[j] = w[j]
            else:
                w[j] = w[j] + 2 * Mu * e_sig[i] * x_fifo[j]

        c_fifo_1[1:] = c_fifo_1[:-1]
        c_fifo_1[0] = c_fifo_2[-1]

        c_fifo_2[1:] = c_fifo_2[:-1]
        c_fifo_2[0] = abs(e_sig[i])

        c_fifo[0] = np.max(c_fifo_1)
        c_fifo[1] = np.max(c_fifo_2)
        c_sig[i] = round(c_fifo[1] - c_fifo[0], error) != 0

        if c_fifo[0] != c_fifo[1] and round(c_fifo[0], error) == round(c_fifo[1], error):
            stabilized = i
            break

    return y_sig, e_sig, c_sig, stabilized


def LMS_Normalized_Float(x_sig, d_sig, Mu, N_LMS: int = 256, error: int = 6, h=None):
    # Initializations
    N = min([len(x_sig), len(d_sig)])
    # LMS Coefficients
    w = np.zeros(N_LMS)
    # Input signal FIFO
    x_fifo = np.zeros(N_LMS)
    # Error signal
    e_sig = np.zeros(N)
    # Output signal
    y_sig = np.zeros(N)

    c_sig = np.zeros(N)
    c_fifo = np.zeros(2)
    c_fifo_1 = np.zeros(N_LMS)
    c_fifo_2 = np.zeros(N_LMS)

    cte = 0.0001

    for i in range(0, N - 1):
        #     # x_fifo[1:] = x_fifo[:-1]
        for j in range(N_LMS - 1):
            x_fifo[N_LMS - 1 - j] = x_fifo[N_LMS - 2 - j]
        x_fifo[0] = x_sig[i]
        x_norm = 0.0
        y_sig_acc = 0
        for j in range(N_LMS):
            y_sig_mul = x_fifo[j] * w[j]
            y_sig_acc += y_sig_mul
            x_norm = x_norm + x_fifo[j] ** 2
        y_sig[i] = y_sig_acc
        e_sig[i] = d_sig[i] - y_sig[i]
        for j in range(N_LMS):
            w[j] = w[j] + 2 * (Mu / (cte + x_norm)) * e_sig[i] * x_fifo[j]

        # Si los coeficients son conocidos
        if h is None:
            c_fifo_1[1:] = c_fifo_1[:-1]
            c_fifo_1[0] = c_fifo_2[-1]

            c_fifo_2[1:] = c_fifo_2[:-1]
            c_fifo_2[0] = abs(e_sig[i])

            c_fifo[0] = np.max(c_fifo_1)
            c_fifo[1] = np.max(c_fifo_2)
            c_sig[i] = round(c_fifo[1] - c_fifo[0], error) != 0

            if c_fifo[0] != c_fifo[1] and round(c_fifo[0], error) == round(c_fifo[1], error):
                break
        else:
            e = 0
            for j in range(N_LMS):
                e = e + abs(w[j] - h[j])
            e = e / N_LMS
            if round(e, error) == 0:
                break
    return y_sig, w, e_sig, i


def filtro_fir(x_sig, h_fir):
    d_sig = np.zeros(len(x_sig))

    # Input signal FIFO
    x_fifo = np.zeros(len(h_fir))

    for i in range(len(x_sig)):
        for j in range(len(h_fir) - 1):
            x_fifo[len(h_fir) - 1 - j] = x_fifo[len(h_fir) - 2 - j]
        # x_fifo[1:] = x_fifo[:-1]
        x_fifo[0] = x_sig[i]
        d_sig_acc = 0
        for j in range(len(h_fir)):
            d_sig_mul = x_fifo[j] * h_fir[j]
            d_sig_acc += d_sig_mul
        d_sig[i] = d_sig_acc
    return d_sig


def sumulation(
    Fs: int,
    N: int,
    repeticiones: int,
    error: int,
    Mu_inicial: float,
    Mu_final: float,
    Mu_paso: float,
    h_fir: list = None,
    N_LMS: int = None,
):
    if h_fir is None:
        tipo = "error"
    else:
        tipo = "fir"
        N_LMS = len(h_fir)

    descripcion = f"Type: {tipo.title()}, Fs:{Fs}, N:{N}, N_LMS:{N_LMS}, R:{repeticiones}, E:{error}, Mu:{Mu_inicial}-{Mu_final} ({Mu_paso})"

    print(f"Starting simulation {descripcion}")
    inicio = time.time()

    n = np.arange(0, N, 1) / Fs
    # Desired signal is a clean sin
    d_sig = np.sin(2 * np.pi * 100 * n)

    # white noise with normal distribution
    media = 0
    sigma = 0.00001
    amplitud = 1 / sigma

    coeficientes = np.arange(Mu_inicial, Mu_final, Mu_paso)
    maximos = np.zeros(len(coeficientes))
    minimos = np.zeros(len(coeficientes))
    promedios = np.zeros(len(coeficientes))

    for corrida in range(repeticiones):
        if h_fir is None:
            # Input signal is the desired sin plus the noise
            x_sig = d_sig + np.random.normal(media, sigma, N)
        else:
            x_sig = amplitud * np.random.normal(media, sigma, N)
            d_sig = filtro_fir(x_sig, h_fir)

        for coeficiente in range(len(coeficientes)):
            Mu = coeficientes[coeficiente]
            _, h, _, indice = LMS_Normalized_Float(x_sig, d_sig, Mu=Mu, error=error, h=h_fir, N_LMS=N_LMS)
            minimos[coeficiente] = min(minimos[coeficiente], indice) if minimos[coeficiente] != 0 else indice
            maximos[coeficiente] = max(maximos[coeficiente], indice)
            promedios[coeficiente] += indice

        duracion = time.time() - inicio
        print(
            f"Iteration {corrida} on simulation {descripcion} completed after {time.strftime('%H:%M:%S', time.gmtime(duracion))}"
        )

    promedios = promedios / repeticiones
    nombre = f"resultados/lms_{tipo}_{N_LMS}_r{repeticiones}_e{error}_mu{Mu_inicial}_{Mu_final}_{Mu_paso}"
    np.savez_compressed(nombre, coeficientes=coeficientes, promedios=promedios, minimos=minimos, maximos=maximos)

    duracion = time.time() - inicio
    print(f"Simulation {descripcion} completed after {time.strftime('%H:%M:%S', time.gmtime(duracion))}")
    return coeficientes, promedios, minimos, maximos


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Firmware del equipo colector de asistencia",
    )
    parser.add_argument(
        "-fs",
        "--frecuencia-muestreo",
        dest="FS",
        type=int,
        default=2000,
        action="store",
        help="Frecuencia de muestreo de la señal",
    )
    parser.add_argument(
        "-n",
        "--muestras",
        dest="N",
        type=int,
        action="store",
        default=200000,
        help="Cantidad de muestras de la señal",
    )
    parser.add_argument(
        "-nlms",
        "--coeficientes",
        dest="N_LMS",
        type=int,
        action="store",
        default=256,
        help="Cantidad de coeficientes en el filtro LMS",
    )
    parser.add_argument(
        "-r",
        "--repeticiones",
        dest="repeticiones",
        type=int,
        action="store",
        default=10,
        help="Cantidad de repeticiones de la simulacion con diferentes señales",
    )
    parser.add_argument(
        "-e",
        "--error",
        dest="error",
        type=int,
        action="store",
        default=3,
        help="Cantidad de decimales en la comparación de convergencia",
    )
    parser.add_argument(
        "-mui",
        "--mu-inicial",
        dest="Mu_inicial",
        type=float,
        action="store",
        help="Valor inicial del coeficiente Mu",
    )
    parser.add_argument(
        "-muf",
        "--mu-final",
        dest="Mu_final",
        type=float,
        action="store",
        help="Valor final del coeficiente Mu",
    )
    parser.add_argument(
        "-mup",
        "--mu-paso",
        dest="Mu_paso",
        type=float,
        action="store",
        help="Valor del paso del coeficiente Mu",
    )
    parser.add_argument(
        "-fir",
        "--filtro-fir",
        dest="fuente",
        # type=string,
        action="store",
        help="Archivo con la definición del filtro a copiar",
    )

    argumentos = parser.parse_args()
    try:
        h_fir = np.load(f"filtros/{argumentos.fuente}.npy")[0].astype(np.float64)
    except:
        h_fir = None

    sumulation(
        Fs=argumentos.FS, N=argumentos.N, repeticiones=argumentos.repeticiones,
        error=argumentos.error, Mu_inicial=argumentos.Mu_inicial, Mu_final=argumentos.Mu_final,
        Mu_paso=argumentos.Mu_paso, N_LMS=argumentos.N_LMS, h_fir=h_fir
    )
