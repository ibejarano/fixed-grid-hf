# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:13:43 2023

@author: YT34520
"""

import numpy as np
import scipy as sp
from math import floor

class CartesianMesh:
    ncycles = 40
    nmin = 60
    nmax = 90
    def __init__(self):
        """
        Parameters
        ----------
        g : float
            Mesh semi-long, dimensionless

        Returns
        -------
        None.

        """
        self.current = self.nmin
        
    def add_element(self):
        self.current += 1
    
    def init_channel(self, n, xf, V, dz):
        self.channel = FluidChannelMesh(n, xf, V, dz)
        self.n = n
    
    def remesh(self):
        #TODO Quedan mas cositas
        raise Exception("Remeshing not implemented yet.")
        self.channel.dz = None
    
    def guess_fluid_front(self, dt):
        gf, mf, phi = self.channel.guess_fluid_front(self.dz, dt, self.n)
        return gf, mf, phi
    
    def set_dz(self, gt0):
        self.dz = gt0 / self.nmin
        
    def get_coordinates(self):
        return self.dz*np.arange(0.5,self.current+0.5,step=1.0)
    
    def update(self, n):
        self.n = n

    
class FluidChannelMesh:
    
    mjump = 10
    
    def __init__(self, n, xft, V, dz):
        self.m = floor(xft*n)
        self.phi = n*xft - self.m
        self.V = V
        self.dz = dz
        self.g = dz*(self.m+ self.phi)
        
    def get_channel_params(self):
        return self.m, self.phi, self.V
        
    def limit_fluidjump(self, mf, n):
        m0 = self.m
        mjump = self.mjump
        
        if mf > (m0+mjump):
            mf = m0 + mjump
        elif mf < (m0 - mjump):
            mf= m0 - mjump
        
        if mf > (n-1):
            mf = n-1
        elif mf < 1:
            mf = 1
        return int(mf)
    
    def guess_fluid_front(self, dz, dt, n):
        gf = (self.m+self.phi)*dz + self.V*dt
        mf = floor(self.g/dz)
        mf = self.limit_fluidjump(mf, n)
        phi = gf/dz - mf
        return gf, mf, phi
    
    def update(self, m, phi, V):
        self.m = m
        self.phi = phi
        self.V = V
        self.g = self.dz*(m+phi)

    