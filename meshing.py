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
        self.current = self.nmin
        
    def add_element(self):
        self.current += 1
    
    def init_channel(self, xf, V):
        dz = self.dz
        n = self.current
        self.channel = FluidChannelMesh(n, xf, V, dz)
    
    def remesh(self, gtLast):
        #TODO Quedan mas cositas
        self.current = self.nmin
        self.set_dz(gtLast)
    
    def guess_fluid_front(self, dt):
        gf, mf, phi = self.channel.guess_fluid_front(self.dz, dt, self.current)
        return gf, mf, phi
    
    def set_dz(self, gt0):
        self.dz = gt0 / self.nmin
        
    def get_coordinates(self):
        return self.dz*np.arange(0.5,self.current+0.5,step=1.0)
        
    def limit_fluidjump(self, mf):
        n = self.current
        
        mf = self.channel.limit_fluidjump(mf, n)
        
        return mf

    
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

    