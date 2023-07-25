# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:14:08 2023

@author: YT34520
"""

import matplotlib.pyplot as plt

def plot_variable_vs_time(time, var, label=None, title=None, ax=None):
    f = None
    if ax == None:
        f, ax = plt.subplots(1)
    
    ax.plot(time, var, label=label)
    plt.xscale("log")
    plt.title(title)
    plt.legend()
    return f, ax