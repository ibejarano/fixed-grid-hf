# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:13:43 2023

@author: YT34520
"""

import numpy as np
import scipy as sp

class Mesh():
    def __init__(self, n0, nmax, ncycles, gt):
        self.n0 = n0
        self.nmax = nmax
        self.ncycles = ncycles
        self.num_remeshing = 0
        self.xi = np.arange(0.5, n0, step=1.0)/self.n0
        self.gt = 0
        
    def add_count(self):
        self.num_remeshing += 1
        
    def remesh():
        pass
    
    def update_elem_size(self):
        self.dz = 0
    
    def get_coordinates(self):
        pass
    
    def get_coordinates_unit(self):
        pass
    
    def interpolate(self, y, xout):
        return so.interpolate.interp1d(self.xi, y)(xout)
    