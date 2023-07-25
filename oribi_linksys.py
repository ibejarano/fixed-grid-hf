# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 10:33:57 2022

@author: YT34520
"""
from math import pi
import numpy as np
from numpy import diag
from scipy.sparse.linalg import LinearOperator, spilu

def build_matrices(Om0, Omk, n, mf, phi, m0, phi0, zeta, dtold, A, D, S, AllinvAlc, vects0, dz, dtdz3, flux, tr):
    d = 2 #Para problema axisym
    Mat = np.zeros((mf, mf))
    Kw = np.zeros((mf, 1))
    Ke = np.zeros((mf, 1))
    Kc = np.zeros((mf, 1))
    
    Kw[1:mf] = (0.5*(Omk[:mf-1] + Omk[1:mf]))**3
    Ke[:mf] = (0.5*(Omk[:mf] + Omk[1:mf+1]))**3
    Kw[1:mf] = Kw[1:mf] * (0.5*(zeta[:mf-1, None] + zeta[1:mf, None]))/zeta[1:mf, None]
    Ke[:mf] = Ke[:mf] * (0.5*(zeta[:mf, None] + zeta[1:mf+1, None]))/zeta[:mf, None]
    
    Kc = Kw+Ke
    Mat[0, :mf] = (-Kc[0]*D[0,:mf] + Ke[0]*D[1,:mf])*dtdz3
    
    # TODO Vectorizar
    for i in range(1, mf-1):
        Mat[i, :mf] = (Kw[i]*D[i-1, :mf] - Kc[i]*D[i, :mf] + Ke[i]*D[i+1, :mf])*dtdz3
    
    Mat[mf-1, :mf] = (Kw[mf-1]*D[mf-2, :mf] - Kw[mf-1]*D[mf-1, :mf] )*dtdz3
    
    G = Mat@Om0[:mf]
    
    G[0] = G[0] + (1+tr)*0.5/(pi*zeta[0])*dtold/dz - (-Kc[0]*S[0]+Ke[0]*S[1])*dtdz3
    for i in range(1, mf-1):
        G[i] = G[i] - (Kw[i]@S[i-1] - Kc[i]@S[i] + Ke[i]@S[i+1])*dtdz3
    
    G[mf-1] = G[mf-1] - flux/(zeta[mf-1]**(d-1))*dtold/dz - (Kw[mf-1]@S[mf-2] - Kw[mf-1]@S[mf-1])*dtdz3
    
    if mf>m0:
        G[m0] = G[m0] - (1- phi0)*Om0[m0]
        if mf > m0+1:
            G[m0+1:mf] = G[m0+1:mf] - Om0[m0+1:mf]
        
    # Solo para solver de punto fijo
    Mat = np.eye(mf) - Mat
    rhs = G
    prec = build_precond(mf, D, Kc, Ke, Kw, dtdz3)
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