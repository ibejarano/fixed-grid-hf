# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:14:32 2022

@author: YT34520
"""

class Secante:
    def __init__(self, x0, ratio, tol=1e-12):
        self.ratio = ratio
        self.x0 = x0
        self.tol = tol
    
    def ite(self, f1):
        try:
            x2 = self.x1 - (self.x1-self.x0)*f1/(f1-self.f0)
            self.x0 = self.x1
            
        except:
            x2 = self.ratio*self.x0
            self.x1 = float("inf")
        
        err = abs(x2 - self.x1)/abs(self.x1)
        self.x1 = x2
        self.f0 = f1

        return x2, (err<self.tol)