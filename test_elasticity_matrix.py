# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:30:32 2022

@author: YT34520
"""

import pytest
from numpy import testing as nptest

# BORRAR!!!
phi = 4.696959e-01
m0 = 4 
phi0 = 2.977845e-01
dtold= 9.000000e-17
dz = 3.468924e-08
dtdz3 = 2.156047e+06
flux = 5.219204e-02 
tr=0

# BORRAR!!!

class TestElasticity:
    
    def test_D_build():
        assert 1 == 2