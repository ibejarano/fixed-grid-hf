# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:28:10 2023

@author: YT34520
"""
import matplotlib.pyplot as plt
from test_numpy_funcs import read_matrix


if __name__ == "__main__":

    # .txt Obtenidos del codigo original para comparar
    tau_ref = read_matrix("tau.txt")
    xft_ref = read_matrix("xft.txt")
    gft_ref = read_matrix("gft.txt")

    plt.plot(tau_ref, xft_ref, "ro", markersize=0.2)
    # plt.ylim(( np.min(problem.xft[:problem.step]),np.max(problem.xft[:problem.step])))
    plt.xlim((tau0, taumax))
    plt.title("Fluid fraction")
    plt.xscale("log")
    plt.show()
