# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 14:16:56 2023

@author: YT34520
"""

import numpy as np

from plot_utilities import plot_variable_vs_time
from read_ini import read_datfile_axisym
from test_numpy_funcs import read_matrix

def calculo_Kstar():
    # K utilizados
    
    Kres = [0.75, 1, 1.25, 1.51, 1.8, 2.15, 1.3]
    tau0 = 1e-16
    
    criterios  = list()
    for i in range(1,8):
        xfref, nref, openings = read_datfile_axisym(f"Oribi_Overtex{i}.dat")
    
        Kstar = Kres[i-1]
        xf_calc = abs(xfref-0.692/0.374*Kstar**(2/3)*tau0**(2/27))/xfref
    
    
        criterios.append(xf_calc)


def get_reference_var(caseNum, var):
    """
    Parameters
    ----------
    num : Int
        Numero de O vertex, desde 1 a 7


    """
    assert caseNum in range(1,8)
    assert var in ["gt", "gft", "xft", "tau"]
    
    SSTARS = [10, 10, 25, 45, 90, 5, 1]
    Sstar = SSTARS[caseNum-1]
    folderDir = f"matlab_reference/overtex{caseNum}_Sstar{Sstar}/"
    
    var_ref = read_matrix(folderDir + var + ".txt")
    
    return var_ref

    
def get_all_references_case(num):
    output_names = ["gt", "gft", "xft", "tau"]
    output = list()
    for i in output_names:
        output.append(get_reference_var(num, i))
        
    return output

