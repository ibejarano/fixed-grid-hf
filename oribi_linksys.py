# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 10:33:57 2022

@author: YT34520
"""
from math import pi
import numpy as np
from numpy import diag
from scipy.sparse.linalg import LinearOperator, spilu

def build_matrices(Om0, Omk, n, dtold, D, S, dz, dtdz3, flux, tr):
    d = 1 #Para problema plano
    Mat = np.zeros((n, n))
    Kw = np.zeros((n, 1))
    Ke = np.zeros((n, 1))
    Kc = np.zeros((n, 1))
    
    OpeningHalfElem = 0.5*(Omk[:-1] + Omk[1:])
    Kw[1:] = OpeningHalfElem**3
    Ke[:-1] = OpeningHalfElem**3
    Kc = Kw+Ke

    Mat[0, :] = (-Ke[0]*D[0,:] + Ke[0]*D[1,:])*dtdz3   
    Mat[1:-1, :] = (Kw[1:-1]*D[:-2, :] - Kc[1:-1]*D[1:-1, :] + Ke[1:-1]*D[2:, :])*dtdz3
    Mat[-1, :] = (Kw[-1]*D[-2, :] - Kw[-1]*D[-1, :] )*dtdz3
    
    G = Mat@Om0[:n]
    
    #TODO Vectorizar
    dtdz2 = dtdz3 * dz
    G[0] = G[0] + (1+tr)*dtold/dz - (-Kc[0]*S[0]+Ke[0]*S[1])*dtdz2
    
    for i in range(1, n-1):
        G[i] = G[i] - (Kw[i]@S[i-1] - Kc[i]@S[i] + Ke[i]@S[i+1])*dtdz2
    
    G[n-1] = G[n-1] - flux*dtold/dz - (Kw[n-1]@S[n-2] - Kw[n-1]@S[n-1])*dtdz2
            
    # Solo para solver de punto fijo
    Mat = np.eye(n) - Mat
    rhs = G
    prec = build_precond(n, D, Kc, Ke, Kw, dtdz3)
    return Mat, rhs, prec

def build_precond(mf, D, Kc, Ke, Kw, dtdz3):
    Dloc =  diag(diag(D, 0))
    Bloc = np.zeros((mf, mf))
    if mf > 1:
        Dloc = Dloc + diag(diag(D,1), 1) + diag(diag(D, -1), -1)
    
    Bloc[0, :mf] = (-Kc[0, None]@Dloc[None, 0, :mf] + Ke[0, None]@Dloc[None, 1, :mf])*dtdz3
    for i in range(1, mf-1):
        Bloc[i, :mf] = (Kw[i, None]@Dloc[None, i-1, :mf] - Kc[i, None]@Dloc[None, i, :mf] + Ke[i, None]@Dloc[None, i+1, :mf])*dtdz3
    
    Bloc[mf-1, :mf] = (Kw[mf-1, None]@Dloc[None, mf-2, :mf] - Kw[mf-1, None]@Dloc[None, mf-1, :mf])*dtdz3
    prec = np.eye(mf) - Bloc
    # If point-fixed
    prec = diag(diag(prec,0)) + diag(diag(prec,1),1) + diag(diag(prec, -1), -1) + diag(diag(prec,2), 2) + diag(diag(prec,-2), -2)
    A_ilu = spilu(prec)
    return LinearOperator( (mf, mf), A_ilu.solve)