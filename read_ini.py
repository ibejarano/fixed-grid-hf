# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 11:49:25 2023

@author: YT34520

Format for data in fileini for d = 2: (Axysimmetric)

 xf
 nref
 Oms(1)
 ...
 Oms(nref)
 
 
"""
def read_datfile_axisym(filepath):
    with open(filepath, "r") as f:
        l = f.readlines()
        xf = float(l[0])
        nref = int(l[1]) 
        openings = [float(i) for i in l[2:]]
    assert nref == len(openings)
    return xf, nref, openings

# Deprecated
def read_datfile_plane_strain(filepath):
    raise Exception("Deprecated")
    with open("Oribi_Overtex1.dat", "r") as f:
        l = f.readlines()
        Kstar = float(l[0])
        gs, gfs, nref = [float(i) for i in l[1].split()]
        openings = [float(i) for i in l[2:]]
    assert nref == len(openings)
    return Kstar, gs, gfs, nref, openings

