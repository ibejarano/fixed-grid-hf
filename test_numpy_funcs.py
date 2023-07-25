# -*- coding: utf-8 -*-

import numpy as np
from numpy import testing as nptest
from unittest import TestCase

from search_root import Secante
from math import cos, pi

class TestSearchRoot(TestCase):

    def test_success_cos_root(self):
        x0 = 1.6
        test_secante = Secante(x0, 1.05)
        x1 = x0
        
        for i in range(50):
            f1 = cos(x1)
            x1, flag = test_secante.ite(f1)
            if flag:
                break
        
        nptest.assert_almost_equal(x1, pi/2)
        
    def test_success_poly_root(self):
        x0 = 1.1
        test_secante = Secante(x0, 1.005)
        x1 = x0
        
        for i in range(100):
            f1 = x1**2 - 1
            x1, flag = test_secante.ite(f1)
            if flag:
                break
        
        nptest.assert_almost_equal(x1, 1)


def read_matrix(fname):
    out = list()
    with open(fname, "r") as f:
        ls = f.readlines()
        for l in ls:
            out.append(float(l.strip()))
    return np.array(out)
            