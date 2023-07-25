# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:25:19 2022

@author: YT34520
"""

import numpy as np
from math import pi
from scipy.special import ellipk, ellipe

# Its not being used yet
class MatricesDD:
    def __init__(self, zeta, n , g, d=2, psi=1):
        #TODO: d=2 axisimetrico
        # psi a definir
        A, Mnn, Mns, Msn, Mss = create_dd_matplane(zeta, n, psi, g)
        alpha = 1/3
        A[n-1, n-1] = A[n-1, n-1] + alpha/4
        self.A = A
        self.Mnn = Mnn
        self.Msn = Msn
        self.Mns = Mns
        self.Mss = Mss

def create_DD_Matrix(zeta, n, g, d=2):
    psi = 0 #FULL PLANE
    A, Mnn, Mns, Msn, Mss = create_dd_axi_full(zeta,n,psi,g)
    A = A/n
    alpha = 1/3 * (8*n - 1)/(8*n-5)
    A[n-1, n-1] = A[n-1, n-1]+alpha/4
    return A, Mnn, Mns, Msn, Mss

def set_mat_values(mat, g_func, n, zeta, up, um, psi):
    mat[:n, n-1] = g_func(zeta, up[n-1], psi) - g_func(zeta, um[n-1], psi)
    mat[n-1, :n-1] = g_func(zeta[n-1], up[:n-1], psi) - g_func(zeta[n-1], um[:n-1], psi)

def update_dd_Matrix(zeta, n, g, dd_mats, psi=1):
    # psi1 = half-plane
    dz = zeta[1] - zeta[0]

    A, Mnn, Mns, Msn, Mss = dd_mats.get_dd_mats()
    up = zeta + 0.5*dz
    um = up - dz
    # HALF PLANE
    set_mat_values(Mss, Gxxl, n, zeta, up, um, psi)
    set_mat_values(Msn, Gyxl, n, zeta, up, um, psi)
    set_mat_values(Mns, Gxyl, n, zeta, up, um, psi)
    set_mat_values(Mnn, Gyyl, n, zeta, up, um, psi)
    x = np.linalg.solve(Mss,Msn)
    A = dz*0.25/pi*(Mnn-Mns@x)

def create_dd_matplane(zeta, n, psi, g):
    dz = zeta[1] - zeta[0]
    # Construir matriz
    up, x = np.meshgrid(zeta+0.5*dz, zeta)
    um = up-dz
    Mss = Gxxl(x,up,psi) - Gxxl(x,um,psi) 
    Msn = Gyxl(x,up,psi) - Gyxl(x,um,psi)
    Mns = Gxyl(x,up,psi) - Gxyl(x,um,psi)
    Mnn = Gyyl(x,up,psi) - Gyyl(x,um,psi)
    # CASO PARTICULAR PSI = 1 Y d=1
    A = dz*0.25/pi*Mnn
    # TERMINA CASO PARTICULAR
    return A, Mnn, Mns, Msn, Mss


def create_dd_axi_full(zeta, n, psi, g):
    zeta = np.expand_dims(zeta, axis=1)
    dz = (zeta[1] - zeta[0])[0]
    zetavectp = 1/(zeta.T + 0.5*dz)
    zetavectm = 1/(zeta.T - 0.5*dz)
    zetavectm[0, 0] = float("NaN")
    zetavect = 1 / zeta.T
    
    # Construir matrices
    zetavp, zetavm = np.meshgrid(zetavectp, zetavectm)
    zetav, aux = np.meshgrid(zetavect, zetavect)
    zetavm = zetavm.T
    zetazetavp = zeta@zetavectp
    zetazetavm = zeta@zetavectm
    
    # psi = 0 full space
    km = k(zetazetavm, np.zeros(zetazetavm.shape))
    kprimem = kprime(zetazetavm, np.zeros(zetazetavm.shape))
    elKm = ellipk(km**2)
    elEm = ellipe(km**2)
    kp = k(zetazetavp, np.zeros(zetazetavp.shape))
    kprimep = kprime(zetazetavp, np.zeros(zetazetavp.shape))
    elKp = ellipk(kp**2)
    elEp = ellipe(kp**2)
    Mnn = -(1/(2*pi))*((-1/4)*zetavp*J101(zetazetavp, np.zeros(zetazetavp.shape), kp, kprimep, elEp, elKp) - (-1/4)*zetavm*J101(zetazetavm, np.zeros(zetazetavm.shape), km, kprimem, elEm, elKm))
    Mnn[:, 0, None] = -(1/(2*pi))*(-1/4)*zetavp[:,0, None]*J101(zetazetavp[:,0, None], np.zeros((n,1)), kp[:,0, None], kprimep[:,0, None], elEp[:,0, None], elKp[:,0, None])
    A = g*Mnn
    Mns = 0
    Msn = 0
    Mss = 0
    
    return A, Mnn, Mns, Msn, Mss 

def create_dd_axi_half(zeta, n, psi, g):
    zeta = np.expand_dims(zeta, axis=1)
    dz = (zeta[1] - zeta[0])[0]
    zetavectp = 1/(zeta.T + 0.5*dz)
    zetavectm = 1/(zeta.T - 0.5*dz)
    zetavectm[0, 0] = float("NaN")
    zetavect = 1 / zeta.T
    
    # Construir matrices
    zetavp, zetavm = np.meshgrid(zetavectp, zetavectm)
    zetav, aux = np.meshgrid(zetavect, zetavect)
    zetavm = zetavm.T
    zetazetavp = zeta@zetavectp
    zetazetavm = zeta@zetavectm
    
    # Psip = 1 HALF-SPACE
    zetap = zeta.T + 0.5*dz
    zetam = zeta.T - 0.5*dz
    
    zetap, zetam = np.meshgrid(zetap, zetam)
    zetam = zetam.T
    zetam[:, 0] = 1.0
    
    Mnn = -(1/(2*pi))*(zetavp * gnnb(zetazetavp, psi*zetap) - zetavm*gnnb(zetazetavm, psi*zetam))    
    Mnn[:, 0] = -(1/(2*pi))*zetavp[:, 0]*gnnb(zetazetavp[:, 0], psi*zetap[:, 0])
    
    Mss = -(1/(2*pi))*(zetav * gssb(zetazetavp, psi*zetap) - zetav*gssb(zetazetavm, psi*zetam))
    Mss[:, 0] = -(1/(2*pi))*zetav[:, 0]*gssb(zetazetavp[:, 0], psi*zetap[:, 0])
    
    Msn = -(1/(2*pi))*(zetavp * gsnb(zetazetavp, psi*zetap) - zetavm*gsnb(zetazetavm, psi*zetam))
    Msn[:, 0, None] = -(1/(2*pi))*zetavp[:, 0, None]*gsnb(zetazetavp[:, 0, None], psi*zetap[:, 0, None])
    
    Mns = -(1/(2*pi))*(zetav * gnsb(zetazetavp, psi*zetap) - zetav*gnsb(zetazetavm, psi*zetam))
    Mns[:, 0] = -(1/(2*pi))*zetav[:, 0]*gnsb(zetazetavp[:, 0], psi*zetap[:, 0])
    
    x = np.linalg.solve(Mss,Msn)
    A = g*(Mnn-Mns@x)

    return A, Mnn, Mns, Msn, Mss

    
def get_mesh(zeta):
    dz = zeta[1] - zeta[0]
    up, x = np.meshgrid(zeta+0.5*dz, zeta)
    return up, x

def gnnb(r, z):
    kv = k(r, 2/z)
    kprimev = kprime(r, 2/z)
    elK = ellipk(kv**2)
    elE = ellipe(kv**2)
    ddgnn = 0.25*J101(r, 2/z, kv, kprimev, elE, elK) + 1/(2*z)*J102(r, 2/z, kv, kprimev, elE, elK) + (1/(2*z**2))*J103(r, 2/z, kv, kprimev, elE, elK)
    
    z = np.zeros(len(r))
    kv = k(r,z)
    kprimev = kprime(r, z)
    elK = ellipk(kv**2)
    elE = ellipe(kv**2)
    gnno = -0.25*(J101(r,z, kv, kprimev, elE, elK))
    
    return gnno + ddgnn

def gssb(r, z):
    kv = k(r,2/z)
    kprimev = kprime(r,2/z)
    elK = ellipk(kv**2)
    elE = ellipe(kv**2)
    ddgss = -(1./4)*J211(r,2/z,kv,kprimev,elE,elK) + 1/(2*z)*J212(r,2/z,kv,kprimev,elE,elK)-(1/(2*z**2))*J213(r,2/z,kv,kprimev,elE,elK)
    z = np.zeros(len(r))
    kv = k(r,z)
    kprimev = kprime(r,z)
    elK = ellipk(kv**2)
    elE = ellipe(kv**2)
    gsso = 0.25*J211(r,z,kv,kprimev,elE,elK)
    return gsso + ddgss

def gsnb(r, z):
    kv = k(r,2/z)
    kprimev = kprime(r,2/z)
    elK = ellipk(kv**2)
    elE = ellipe(kv**2)
    return (1/(2*z**2))*J113(r,2/z,kv,kprimev,elE,elK)

def gnsb(r, z):
    kv = k(r, 2/z)
    kprimev = kprime(r,2/z)
    elK = ellipk(kv**2);
    elE = ellipe(kv**2)
    return -(1/(2*z**2))*J203(r,2/z,kv,kprimev,elE,elK)

def k(r, z):
    return np.sqrt(4*r/((1+r)**2 + z**2))

def kprime(r, z):
    return np.sqrt(((1-r)**2 + z**2)/((1+r)**2 + z**2))

def J101(r, z, k, kprime, elE, elK):
    return k*r**(-0.5)*(elE*k**2*(1-r**2 - z**2)/(4*r*kprime**2)+elK)

def J102(r, z, k, kprime, elE, elK):
    t1 = (z*k**3)/((4*kprime**2)*(r**(3/2)))
    t2 = (((( k**4 )*(1-(r**2+z**2)**2))/(4*r*r*kprime*kprime)+3)*elE + elK*k*k*(r*r + z*z -1)/(4*r))
    return t1*t2

def J103(r, z, k, kprime, elE, elK):
    t1 = (k**3)/((4*kprime**2)*(r**(3/2)))
    a = (((k**4)*(1-(r**2 + z**2)**2))/(4*r**2*kprime**2)+3)
    b = (k**2*z**2*(2-k**2)/(2*kprime**2*r)-1)
    c = (z**2*k**4)/(16*r**2*kprime**2) * (2*k**2*(2-k**2)*(1-(r**2+z**2)**2)/(kprime**2*r) + 17*(r**2+z**2)-1)
    t2 = (a*b+c)*elE
    t3 = (k**2)/(4*r)*(1-r**2-2*z**2*((k**4)*(1-(r**2+z**2)**2)/(4*r**2*kprime**2)+3))*elK
    return t1*(t2+t3)

def J211(r,z,k,kprime,elE,elK):
    return 2*J110(r,z,k,kprime,elE,elK)-J011(r,z,k,kprime,elE,elK)

def J110(r,z,k,kprime,elE,elK):
    return 4/(k*r**(1/2))*((1-(1/2)*k**2)*elK - elE)

def J011(r,z,k,kprime,elE,elK):
    return elE*(k**3)*(r**2-1-z**2)/((4*kprime**2)*(r**(5/2))) + k*elK/(r**(3/2))

def J212(r,z,k,kprime,elE,elK):
    return 2*J111(r,z,k,kprime,elE,elK)-J012(r,z,k,kprime,elE,elK)

def J111(r,z,k,kprime,elE,elK):
    return k*z*r**(-3/2)*((2-k**2)/(2*kprime**2)*elE - elK)

def J012(r,z,k,kprime,elE,elK):
    return k**3*z/(4*kprime**2*r**(5/2))*((k**4*(r**4-(1+z**2)**2)/(4*r**2*kprime**2)+3)*elE + elK*k**2*(1-r**2+z**2)/(4*r))

def J213(r,z,k,kprime,elE,elK):
    return 2*J112(r,z,k,kprime,elE,elK)-J013(r,z,k,kprime,elE,elK)

def J112(r,z,k,kprime,elE,elK):
    return k/r**(3/2)*(k**2/(4*r*kprime**2)*((k**4*z**2)/kprime**2-1-r**2)*elE + (1-(k**2*z**2*(2-k**2))/(8*r*kprime**2))*elK)

def J013(r,z,k,kprime,elE,elK):
    return k**3/(4*kprime**2*r**(5/2))*(((k**4*(r**4-(1+z**2)**2)/(4*r**2*kprime**2)+3)*(k**2*(2-k**2)*z**2/(2*kprime**2*r)-1) + k**4*z**2/(16*kprime**2*r**2)*(2*k**2*(2-k**2)*(r**4-(1+z**2)**2)/(kprime**2*r)+17*(1+z**2)-r**2))*elE + k**2/(4*r)*(r**2-1-2*z**2*(k**4*(r**4-(1+z**2)**2)/(4*kprime**2*r**2)+3))*elK)

def J203(r,z,k,kprime,elE,elK):
    return 2*J102(r,z,k,kprime,elE,elK)-J003(r,z,k,kprime,elE,elK)

def J003(r,z,k,kprime,elE,elK):
    return (k**5*z)/(32*kprime**4*r**(5/2))*((12*kprime**2-(4*k**2*z**2*(1+kprime**2))/r)*elK - (24*(1+kprime**2)-(k**2*z**2*(8+7*kprime**2+8*kprime**4))/(r*kprime**2))*elE)

def J113(r,z,k,kprime,elE,elK):
    return (k**3*z)/(8*kprime**4*r**(5/2))*(((k**2*z**2*(2-k**2)*(8*k**4+3*kprime**2))/(4*r*kprime**2) -6*(1-k**2*kprime**2))*elE + k**2/(2*r)*(3*kprime**2*(1+r**2)-2*k**4*z**2)*elK)


def Gxxl(x, u, gam):
    g1 = 1+0.25*gam**2*(x-u)**2
    g2 = 1+0.25*gam**2*(x-u)**2
    res = 1/(u-x)-1.0/(u+x)-gam**2.0*(u-x)*(g1**(-3.0)-0.75*g1**(-2.0)+0.25*g1**(-1.0))
    res = res+gam**2*(u+x)*(g2**-3 - 0.75*g2**-2 + 0.25*g2**-1)
    return res

def Gyxl(x, u, gam):
    g1=1.0+0.25*gam**2*(x-u)**2
    g2=1.0+0.25*gam**2*(x+u)**2
    return gam*(-2.0*g1**-3 + 1.5*g1**-2 + 2.0*g2**-3 - 1.5*g2**-2.0)

def Gxyl(x, u, gam):
    g1 = 1.0 + 0.25*gam**2*(x-u)**2
    g2 = 1.0 + 0.25*gam**2*(x+u)**2
    return gam*(2*g1**-3 - 1.5*g1**-2 + 2*g2**-3 - 1*g2**2)

def Gyyl(x, u, gam):
    if gam == 0:
        raise
    else:
        g1 = 1 + 0.25*gam**2*(x-u)**2
        g2 = 1 + 0.25*gam**2*(x+u)**2
        res = 1/(u-x) + 1/(u+x) - gam**2*(u-x)*(g1**-3 + 0.25*g1**-2+0.25*g1**-1)
        res = res - gam**2*(u+x)*(g2**-3 + 0.25*g2**-2 + 0.25*g2**-1)
        return res

if __name__ == "__main__":
    
    def get_coordinates(dz, n):
        return dz*np.arange(0.5,n+0.5,step=1.0)
    
    n = 11
    dz = 1.040677216425466e-07
    g = 1.144744938068012e-06
    zeta = get_coordinates(dz, n)
    psi = 1
    A, Mnn, Mns, Msn, Mss = create_DD_Matrix(zeta, n, g)
    A_ref = read_matrix("A.txt")
    Mnn_ref = read_matrix("Mnn.txt")
    print(A - A_ref)