# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 14:10:25 2022

@author: YT34520
"""

import numpy as np
import matplotlib.pyplot as plt

with open("ForceVcDisp.txt", "r") as f:
    ls= f.readlines()
    out = list()
    for l in ls:
        out.append([float(x) for x in l.strip().split()])
        

arr = np.array(out)

plt.plot(arr[:,0], arr[:, 1])
plt.ylabel("Fuerza Reacción Y [N]")
|plt.xlabel("Desplazamiento Y [mm]")