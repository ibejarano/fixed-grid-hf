# -*- coding: utf-8 -*-

from math import floor
import numpy as np
from scipy.interpolate import interp1d

from read_ini import read_datfile_axisym


class AxisymInitializer:
    
    def __init__(self, tau0, name=None, datPath=None):
        self.tau0 = tau0
        self.initFilename = name
        
    def compute_initial_values(self, Kstar):
        tau0 = self.tau0
        gs = 0.374*Kstar**(-2/3)
        gfs = 0.692
        alpg = 10/27
        alpgf = 4/9 
        
        gt0  = gs*tau0**alpg
        gft0 = gfs*tau0**alpgf 
        Vtipt0 = gs*alpg*tau0**(alpg-1)
        Vt0 = gfs*alpgf*tau0**(alpgf-1)        
        xft0 = gft0/gt0
    
        return gt0, gft0, Vtipt0, Vt0, xft0
    
    def compute_initial_mesh_values(self, mesh, gt0, xft0):
        alpo = 1/9
        n0 = mesh.nmin
        Om = np.zeros((mesh.nmax,1))
        mf0 = floor(xft0*n0)
        phit0 = n0*xft0-mf0
        xi = np.arange(0.5, n0+0.5, step=1.0)/n0
        xfref, nref, openings_ini = read_datfile_axisym(self.initFilename)    
        xiref = np.arange(0.5, nref+0.5, step=1.0)/nref
        Om[:n0, 0] = interp1d(xiref, openings_ini, kind="cubic")(xi)*self.tau0**alpo
        
        return Om
    
    def compute_initial_dt(self, dz, Vtip):
        return dz / Vtip