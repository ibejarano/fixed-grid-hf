import numpy as np
import matplotlib.pyplot as plt


def dimensionalize():
    # Parametros

    d = 2 # Axisimetrico

    K1cprime = 1e6
    Eprime = 3e9
    muprime = 1e-3 
    Qo = 1e-6 / 60
    Sigmao = 2.2e5
    H = 22/1000


    Kstar = K1cprime / (Qo * Eprime**3 * muprime * H**(1-d))**(1/4)
    Sstar = Sigmao * H **((1+d) / 4) / (Qo * Eprime**3 * muprime)**(1/4)


    tau0 = 1e-16
    xfref = np.linspace(0.5, 1, 20)
    criterion = abs(xfref-0.692/0.374*Kstar**(2/3)*tau0**(2/27))/xfref
    # Xref = 0.57
    
    Tdim = (muprime * H**(1+d) / (Eprime * Qo**3))**(1/4)

    return Tdim, H