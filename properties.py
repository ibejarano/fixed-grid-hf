# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 08:35:44 2023

@author: YT34520
"""
from math import floor, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import bicgstab

from oribi_dd_matrices import create_DD_Matrix
from oribi_linksys import build_matrices
from fracture_initializer import AxisymInitializer
from search_root import Secante
from meshing import CartesianMesh

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
    
    def __init__(self, name, finalTIme):
        self.name = name
        self.finalTime = finalTime

class FractureState:
    
    def __init__(self, g, gf, t, xf, n, m, phi, dt, dz):
        self.g = g
        self.gf = gf
        self.t = t
        self.xf = xf
        self.n = n
        self.m = m
        self.phi = phi
        self.dt = dt
        self.dz = dz
        
    def get_previous_fluid_front(self):
        gf = (m0+phi0)*self.dz + V*self.dtold

        return self.m0, self.V, self.phi
        

class Fracture:
                
    # Iteration tolerances
    iteMaxOpening , iteMaxTimeStep, iteMaxFluidFront = 10, 6, 10
    # Error tolerances
    tolOpening, tolTimeStep, tolFluidFront = 1e-4, 1e-4, 1e-3
    # Time parameters
    taumax = 1e-4
    # mesh extension
    maxCycles = 40
    
    DTRATIO = 1.05
    

    def __init__(self, mesh=None, initializer=None, solid=None, fluid=None, injection=None, Kstar=None, Sstar=None):
        assert mesh, "Mesh not defined"
        assert initializer, "Fracture initializer not defined"
        self.mesh = mesh
        self.initializer = initializer
        
        if solid and fluid and injection:
            self.compute_dimensional_parameters()
        else:
            assert Kstar and Sstar, "Problem undefined."
            self.Kstar = Kstar
            self.Sstar = Sstar
    
    def get_opening(self, offset=0):
        return self.Opening
    
    def set_opening(self, Op):
        self.Opening = np.zeros((self.mesh.nmax, 1))
        self.Opening[:self.mesh.current] = Op[:self.mesh.current]

    def initialize_values(self):
        gt0, gft0, Vtipt0, Vt0, xf0 = self.initializer.compute_initial_values(self.Kstar)
        self.mesh.set_dz(gt0)
        self.mesh.init_channel(self.mesh.nmin, xf0, Vt0, self.mesh.dz)
        self.Omtip = (2/3)*self.Kstar*self.mesh.dz**0.5
        O0 = self.initializer.compute_initial_mesh_values(self.mesh, gt0, xf0)
        self.dtOld = self.initializer.compute_initial_dt(self.mesh.dz, Vtipt0)
        self.set_opening(O0)
        return O0
    
    def compute_dimensional_parameters(self):
        # Con las properties computar los adimensionales
        Kstar = 33.2
        Sstar = 44.22
        self.Kstar = Kstar
        self.Sstar = Sstar
        
    def store_values(self):
        
        # Guardar solamente
        # xft, gt, ,tau
        # gft sale de calculo
        # guardar el ultimo paso nada mas, no crear vectores
        pass
        
    def print_summary(self):
        print("NOT implemented")
        
    def solve_step(self, Op_0, Op_0prev=None):
        dz = self.mesh.dz
        self.mesh.add_element()
        n = self.mesh.current
        gt = n*dz

        Om_k, domold = self.guess_opening(Op_0, n) #TODO: Falta el prev prev

        zeta = self.mesh.get_coordinates()
        A, Mnn, Mns, Msn, Mss = create_DD_Matrix(zeta, n, gt) #TODO 
    
        m, phi, V, dt = self.search_fluid_front(Om_k, n, A, domold)
        
        # Update values
        self.mesh.update(n)
        self.mesh.channel.update(g, m, phi, V)
    
    def search_fluid_front(self, Om_k, n, A, domold):
        gf0, m0, phi0 = self.mesh.channel.get_channel_params()
        gf_k, m_k, phi_k = self.mesh.guess_fluid_front(self.dtOld)
        dz = self.mesh.dz
        dt0 = self.dtOld
        V0 = self.mesh.channel.V
        converged = False       
        for k in range(0, self.iteMaxFluidFront):
            
            # !TODO: Refactorear timestep, recibe muchisimo
            dtnew = self.search_timestep(dt0, m_k, A, domold, phi_k, Om_k)
            # Buscamos la velocidad del frente de fluido
            Pc = A[:m_k, :n]@Om_k[:n]/dz
            V_k = self.compute_fluid_velocity(Pc, m_k, phi_k, Om_k)
            V_k = (V_k + V0)/2 # Promediamos
            # Buscamos el frente de fluido
            gfnew = (m0+phi0)*dz + V_k*dtnew
            m_k = floor(gfnew/dz)
            phi_k = gfnew/dz - m_k
            m_k = self.mesh.limit_fluidjump(m_k, n)
            err = self.compute_error(gf_k, gfnew)
            gf_k = gfnew 
            
            converged = err < self.tolFluidFront
            if converged:
                break
            
        if not converged:
            print("NOT CONVERGED:: MAX ITERATIONS REACHED")
        
        return m_k, phi_k, V_k, dtnew
    
    def search_timestep(self, dt0, m_k, A, domold, phi_k, Om_k):
        n = self.mesh.current

        k2 = 0
        converged = False
        Om_k[n-1] = self.Omtip
        search_root = Secante(dt0, self.DTRATIO, tol=self.tolTimeStep)
        while (k2 < self.iteMaxTimeStep) and (not converged):
            Om_k[n-1] = self.Omtip
            Om_k = self.search_opening(m_k, Om_k, A, domold, dt0, phi_k)
            F1 = Om_k[n-1, 0] - self.Omtip
            dtnew, converged = search_root.ite(F1)
            dtold = dtnew
            k2 += 1
        del search_root
        return dtnew
    
    def search_opening(self, m_k, Om_k, A, domold, dt0, phi_k):
        converged = False
        O_prev = self.get_opening(offset=1)
        tr = 0 #TODO no implementado aun
        domnew = domold.copy()
        n = self.mesh.current
        dz = self.mesh.dz
        
        dtdz3 = dt0 / (dz**3)
        _, AllinvAlc, vecs0, D, S = self.construct_components_elasticity(m_k, A)
        
        m0, phi0, _ = self.mesh.channel.get_channel_params()
        zeta = self.mesh.get_coordinates()
        
        for ite in range(self.iteMaxOpening):
            flux = phi_k * Om_k[m_k, 0]
        
            if m_k > 1:
                Mat, rhs, prec = build_matrices(O_prev, Om_k, n, m_k, phi_k, m0, phi0, zeta, dt0, A, D, S, AllinvAlc, vecs0, self.mesh.dz, dtdz3, flux, tr)
                
                res, flag = bicgstab(Mat, rhs, tol=self.tolOpening*10**(-3), atol=self.tolOpening*10**(-3), M=prec, x0=domold[:m_k], maxiter=100)
                domnew[:m_k] = np.expand_dims(res, axis=1)
            elif m_k == 1:
                zeta = self.get_coordinates(n)
                domnew[0] = ((1+tr)*0.5/pi - flux)*dt0/(dz*(zeta[0]))
            else:
                raise
                
            # Update opening in channel
            Om_k[:m_k] = O_prev[:m_k] + domnew[:m_k]
            # Update opening in the lag
            Om_k[m_k:n] = -AllinvAlc@Om_k[:m_k] - vecs0
            domnew[m_k:n] = Om_k[m_k:n] - O_prev[m_k:n]
            
            # CHECK ERROR 
            err = sqrt(np.sum((domnew-domold)**2))/sqrt(np.sum(domnew**2))
            
            if err < self.tolOpening:
                converged = True
        return Om_k
    
    def compute_fluid_velocity(self, Pc, mf, phi, Omk):
        Pm = Pc[mf-1, 0]
        Vint = 0
        if mf > 1:
            Pmm1 = Pc[mf-2, 0]
            Pmp1 = Pm - (Pm + self.Sstar) / (0.5+phi)
            Vint = -Omk[mf-1, 0]**2*(Pmp1 - Pmm1)/(2* self.mesh.dz)        
        elif mf == 1:
            gradPlin = -(Pm+ self.Sstar)/((0.5+phi)* self.mesh.dz)
            Omint = 0.5*(Omk[mf-1,0]+Omk[mf, 0])
            Vint = -Omint**2*gradPlin
        else:
            raise Exception("mf is negative")

        return Vint

    def construct_components_elasticity(self, mf, A):
        '''
        Ecuaciones (40) - (42) del paper de Gordeily y Detournay
        dz (float)
        Sstar (float)
        n (int)
        mf (int)
        A (numpy matrix) nxn 
        '''
        n = self.mesh.current
        dz = self.mesh.dz
        invA_lag = np.linalg.inv(A[mf:n, mf:n])
        AllinvAl = invA_lag@A[mf:n, :mf]
        vects = dz*self.Sstar*np.sum(invA_lag, axis=1, keepdims=True)
        D = A[:mf, :mf] - A[:mf, mf:n]@AllinvAl
        S = A[:mf, mf:n]@vects
        return invA_lag, AllinvAl, vects, D, S
    
    def run(self):
        taumax = float(self.taumax)
        step = 0
        Op_0 = self.initialize_values()
        
        #for cycle in range(self.maxCycles):
        for n in range(self.mesh.nmin+1, self.mesh.nmax+1):
            step += 1
            
            self.solve_step(Op_0)
            #if self.tau > self.taumax:
                #break
            
            if step % 1:
                print("Step:", step)
        # problem.remesh()
        # plt.show()
            
    def guess_opening(self, Oprev, n, Oprevprev=None):
        dO_old = np.zeros((n,1))
        O_k = Oprev.copy()

        if Oprevprev:
            dO_old[1:n, 0] = Oprev[:n-1] - Oprevprev[:n-1]
            dO_old[0, 0] = dO_old[1, 0]
        O_k[ :n] += dO_old
        O_k[n-1, 0] = self.Omtip
        return O_k, dO_old
            
mesh = CartesianMesh()
initializer = AxisymInitializer(tau0=1e-16, name="Oribi_Overtex1.dat")
fracture = Fracture(mesh=mesh,initializer=initializer, Kstar=1, Sstar=1)
fracture.run()