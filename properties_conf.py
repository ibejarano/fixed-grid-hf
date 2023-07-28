# -*- coding: utf-8 -*-

import os
import numpy as np
import logging
logging.basicConfig(format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)


from datetime import datetime

class FluidProperties:
    
    def __init__(self, viscosity):
        self.viscosity = viscosity
    
class MaterialProperties:
    
    def __init__(self, Eprime, Kprime, sigma0):
        pass
    
    
    def compute_opening(self):
    
        pass

class InjectionProperties:
    
    def __init__(self, Qo=None):
        pass
    
    
    def compute_opening(self):
        pass
    
    
class SimulationProperties:
    
    ROOT_SAVE_PATH = "__simData__"
    
    iteMaxOpening , iteMaxTimeStep, iteMaxFluidFront = 10, 6, 10
    # Error tolerances
    tolOpening, tolTimeStep, tolFluidFront = 1e-4, 1e-4, 1e-3
    # Time parameters
    taumax = 1e-4
    # mesh extension
    maxCycles = 40
    # 
    DTRATIO = 1.05
    # sim date
    simDate = str(datetime.now()).replace(" ", "_").replace(":", "").split(".")[0]
    
    def __init__(self, name=None):
        self.name = self.simDate
        self.savePath = os.path.join(self.ROOT_SAVE_PATH, self.name)

    def set_simulation_name(self, name):
        self.name = name.replace(" ", "_")
        self.savePath = os.path.join(self.ROOT_SAVE_PATH, self.name)