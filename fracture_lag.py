# -*- coding: utf-8 -*-
import logging
import numpy as np
import csv
import os
from math import floor, sqrt, pi
from scipy.sparse.linalg import bicgstab

from oribi_dd_matrices import create_DD_Matrix
from oribi_linksys import build_matrices
from search_root import Secante


def load_fracture_results(path=None, name=None):
    if path:
        raise Exception("Not implemented")
    elif name:
        PATH = "__simData__"
        out = list()
        with open(os.path.join(PATH, name, "data") + ".csv", "r", newline="") as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                out.append([float(d) for d in row])
            
        return np.array(out)
            
    else:
        raise Exception("Fracture name not specified")

class Fracture:
    
    def __init__(self,mesh=None, initializer=None, solid=None, fluid=None, injection=None, simProps=None, Kstar=None, Sstar=None):
        assert mesh, "Mesh not defined"
        assert initializer, "Fracture initializer not defined"
        self.mesh = mesh
        self.initializer = initializer
        self.simProps = simProps
        
        if solid and fluid and injection:
            self.compute_dimensional_parameters()
        else:
            assert Kstar and Sstar, "Problem undefined."
            self.Kstar = Kstar
            self.Sstar = Sstar
            
        self.state = []
    
    def get_opening(self, offset=0):
        return self.Opening
    
    def set_opening(self, Op):
        self.Opening = np.zeros((self.mesh.nmax, 1))
        self.Opening[:self.mesh.current] = Op[:self.mesh.current]

    def initialize_values(self):
        gt0, gft0, Vtipt0, Vt0, xf0 = self.initializer.compute_initial_values(self.Kstar)
        self.mesh.set_dz(gt0)
        self.mesh.init_channel(xf0, Vt0)
        self.Omtip = (2/3)*self.Kstar*self.mesh.dz**0.5
        O0 = self.initializer.compute_initial_mesh_values(self.mesh, gt0, xf0)
        self.dtOld = self.initializer.compute_initial_dt(self.mesh.dz, Vtipt0)
        self.set_opening(O0)
        self.lastTau = self.initializer.tau0
        return O0

    def remesh(self):
        g = self.compute_fracture_length()
        self.mesh.remesh(g)

        
    
    def compute_dimensional_parameters(self):
        raise Exception("Not implemented")
    
    def compute_fracture_length(self):
        return self.mesh.current * self.mesh.dz
    
    def store_values(self):
        g = self.compute_fracture_length()
        time = self.lastTau
        self.state.append([self.step, time, g])
        
    def save_to_file(self):
        savePath = self.simProps.savePath

        if not os.path.exists(self.simProps.ROOT_SAVE_PATH):
            os.mkdir(self.simProps.ROOT_SAVE_PATH)

        if not os.path.exists(savePath):
            os.mkdir(savePath)
            print("Directory: ", savePath, "created")

        with open(os.path.join(savePath, 'data') + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.state)
        
    def print_summary(self):
        print("Simulation results")
        print("Converged steps: ", self.step)
        print("Tau min", self.initializer.tau0, "Tau max", self.lastTau)
        
    def solve_step(self, Op_0, Op_0prev=None):
        self.mesh.add_element()
        gt = self.compute_fracture_length()
        Om_k, domold = self.guess_opening(Op_0, Op_0prev) #TODO: Falta el prev prev

        zeta = self.mesh.get_coordinates()
        A, Mnn, Mns, Msn, Mss = create_DD_Matrix(zeta, self.mesh.current, gt) #TODO 
        m, phi, V, dt = self.search_fluid_front(Om_k, A, domold)
        
        # Update values
        self.mesh.channel.update(m, phi, V)
        self.lastTau += dt
        self.dtOld = dt
        self.store_values()
        
        return Om_k
    
    def search_fluid_front(self, Om_k, A, domold):
        m0, phi0, V0 = self.mesh.channel.get_channel_params()
        gf_k, m_k, phi_k = self.mesh.guess_fluid_front(self.dtOld)
        dz = self.mesh.dz
        n = self.mesh.current
        converged = False       
        for k in range(0, self.simProps.iteMaxFluidFront):
            # !TODO: Refactorear timestep, recibe muchisimo
            dtnew = self.search_timestep(m_k, A, domold, phi_k, Om_k)
            # Buscamos la velocidad del frente de fluido
            Pc = A[:m_k, :n]@Om_k[:n]/dz
            V_k = self.compute_fluid_velocity(Pc, m_k, phi_k, Om_k)
            V_k = (V_k + V0)/2 # Promediamos
            # Buscamos el frente de fluido
            gfnew = (m0+phi0)*dz + V_k*dtnew
            m_k = floor(gfnew/dz)
            phi_k = gfnew/dz - m_k
            m_k = self.mesh.limit_fluidjump(m_k)
            err = self.compute_error(gf_k, gfnew)
            gf_k = gfnew 
            
            converged = err < self.simProps.tolFluidFront
            if converged:
                break
            
        if not converged:
            print("NOT CONVERGED:: MAX ITERATIONS REACHED")
        
        return m_k, phi_k, V_k, dtnew
    
    def search_timestep(self, m_k, A, domold, phi_k, Om_k):
        n = self.mesh.current
        k2 = 0
        converged = False
        dt0 = self.dtOld
        search_root = Secante(dt0, self.simProps.DTRATIO, tol=self.simProps.tolTimeStep)
        
        while (k2 < self.simProps.iteMaxTimeStep) and (not converged):
            Om_k[n-1] = self.Omtip
            Om_k = self.search_opening(m_k, Om_k, A, domold, dt0, phi_k)
            F1 = Om_k[n-1, 0] - self.Omtip
            dtnew, converged = search_root.ite(F1)
            dt0 = dtnew
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
        
        for ite in range(self.simProps.iteMaxOpening):
            flux = phi_k * Om_k[m_k, 0]
        
            if m_k > 1:
                Mat, rhs, prec = build_matrices(O_prev, Om_k, n, m_k, phi_k, m0, phi0, zeta, dt0, A, D, S, AllinvAlc, vecs0, self.mesh.dz, dtdz3, flux, tr)
                
                res, flag = bicgstab(Mat, rhs, tol=self.simProps.tolOpening*10**(-3), atol=self.simProps.tolOpening*10**(-3), M=prec, x0=domold[:m_k], maxiter=100)
                domnew[:m_k] = np.expand_dims(res, axis=1)
            elif m_k == 1:
                zeta = self.mesh.get_coordinates()
                domnew[0] = ((1+tr)*0.5/pi - flux)*dt0/(dz*(zeta[0]))
            else:
                raise
                
            # Update opening in channel
            Om_k[:m_k] = O_prev[:m_k] + domnew[:m_k]
            # Update opening in the lag
            Om_k[m_k:n] = -AllinvAlc@Om_k[:m_k] - vecs0
            domnew[m_k:n] = Om_k[m_k:n] - O_prev[m_k:n]
            
            err = sqrt(np.sum((domnew-domold)**2))/sqrt(np.sum(domnew**2))
            domold[:] = domnew[:]
            if err < self.simProps.tolOpening:
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
        logging.info("Running simulation...")
        self.step = 0
        Op_0 = self.initialize_values()
        Op_pp = None
        for cycle in range(self.simProps.maxCycles):
            for n in range(self.mesh.nmin+1, self.mesh.nmax+1):
                self.step += 1
                
                Op_h = self.solve_step(Op_0, Op_pp)
                
                if Op_pp is not None:
                    Op_pp[:] = Op_0[:]
                else:
                    Op_pp = Op_0.copy()
                Op_0[:] = Op_h[:]
                
                if self.lastTau > self.simProps.taumax:
                    brek
                logging.warning(f"Step: {step}")
            self.remesh()
            logging.warning("REMESH")
        self.save_to_file()
        self.print_summary()
        
    def compute_error(self, val, newval):
        if np.linalg.norm(val) > 0.0:
            err = abs(newval - val)/abs(val)
        else:
            err = np.linalg.norm(newval-val)/np.linalg.norm(newval)
        return err
            
    def guess_opening(self, Oprev, Oprevprev=None):
        n = self.mesh.current
        dO_old = np.zeros((n,1))
        O_k = Oprev.copy()

        if Oprevprev is not None:
            dO_old[1:n, 0] = Oprev[:n-1, 0] - Oprevprev[:n-1, 0]
            dO_old[0, 0] = dO_old[1, 0]
        O_k[ :n] += dO_old
        O_k[n-1, 0] = self.Omtip
        return O_k, dO_old