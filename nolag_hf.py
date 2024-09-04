from math import floor, pi, sqrt
import numpy as np
from scipy.sparse.linalg import bicgstab
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import yaml

from oribi_dd_matrices import create_DD_Matrix
from oribi_linksys import build_matrices

from search_root import Secante
from read_ini import read_datfile_axisym
from oribi_results_reference import get_all_references_case
from dimensionalize import dimensionalize

with open("solver_defaults.yml", "r") as stream:
    try:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)
        

def limit_fluidjump(mf, m0, n, mjump):
    if mf > (m0+mjump):
        mf = m0 + mjump
    elif mf < (m0 - mjump):
        mf= m0 - mjump
    
    if mf > (n-1):
        mf = n-1
    elif mf < 1:
        mf = 1
    return int(mf)

def construct_components_elasticity(n, mf, A, dz, Sstar):
    '''
    Ecuaciones (40) - (42) del paper de Gordeily y Detournay
    dz (float)
    Sstar (float)
    n (int)
    mf (int)
    A (numpy matrix) nxn 
    '''
    invA_lag = np.linalg.inv(A[mf:n, mf:n])
    AllinvAl = invA_lag@A[mf:n, :mf]
    vects = dz*Sstar*np.sum(invA_lag, axis=1, keepdims=True)
    D = A[:mf, :mf] - A[:mf, mf:n]@AllinvAl
    S = A[:mf, mf:n]@vects
    return invA_lag, AllinvAl, vects, D, S

        
class FHProblem:
    def __init__(self, data):
        k1max, k2max, k3max = data.get("max_iterations").values()
        self.k1max , self.k2max, self.k3max = int(k1max), int(k2max), int(k3max)
        eps1, eps2, eps3 = data.get("tolerances").values()
        self.eps1, self.eps2, self.eps3 = float(eps1), float(eps2), float(eps3)
        self.tau0, self.taumax, self.DTRATIO = data.get("time_parameters").values()
        self.n0, self.nmax, self.ncycles, self.mjump = data.get("mesh").values()
        self.Sstar, self.Kstar = data.get("parameters").values()
        dat_filepath = data.get("dat_filepath")
        mmax = (self.nmax-self.n0)*self.ncycles + 1
        nmax = self.nmax
        self.mmax = mmax
        self.Om = np.zeros((nmax,mmax))
        self.Pr = np.zeros((nmax,mmax))- self.Sstar
        self.tau = np.zeros((mmax), dtype=np.float32)
        self.gt = np.zeros((mmax))
        self.gft = np.zeros((mmax))
        self.phit = np.zeros((mmax))
        self.nt = np.zeros(mmax, dtype=np.int32)
        self.mft = np.zeros(mmax, dtype=np.float32)
        self.Vt = np.zeros((mmax))
        self.Vtipt = np.zeros((mmax,1))
        self.xft = np.zeros((mmax,1))
        self.vol = np.zeros((mmax,1))
        self.errvol = np.zeros((mmax,1))
        self.errtip = np.zeros((mmax,1))
        self.step = 0
        
        self.interpolate_initial_solution(dat_filepath)
    
    def interpolate_initial_solution(self, filepath):
        n0 = self.n0
        tau0 = float(self.tau0)
        xfref, nref, openings_ini = read_datfile_axisym(filepath)
        # Para plane strain
        if abs(xfref-0.692/0.374*self.Kstar**(2/3)*tau0**(2/27))/xfref > 0.1:
            print("WARNING: FLUID FRACTION XF IN FILEINI DOES NOT MATCH SOLUTION AT INITIAL TIME")
            
        gs = 0.374*self.Kstar**(-2/3)
        gfs = 0.692
        alpg = 10/27
        alpgf = 4/9 
        alpo = 1/9
        self.tau[0]= tau0
        self.nt[0]= n0
        self.gt[0]  = gs*self.tau[0]**alpg
        self.gft[0] = gs*self.tau[0]**alpgf
        self.Vtipt[0] = gs*alpg*self.tau[0]**(alpg-1)
        self.Vt[0]    = gfs*alpg*self.tau[0]**(alpg-1)           
        self.xft[0] = 1
        self.mft[0] = self.nt[0]
        self.phit[0] = 0.0
        xi = np.arange(0.5, self.nt[0]+0.5, step=1.0)/self.nt[0]
        xiref = np.arange(0.5, nref+0.5, step=1.0)/nref
        self.Om[: self.nt[0], 0] = interp1d(xiref, openings_ini, kind="cubic")(xi)*self.tau[0]**alpo
        self.dz = self.gt[0]/self.nt[0]
        self.dtold = self.dz/ self.Vtipt[0,0]
        
    def solve_step(self, n):
        assert hasattr(self, "dz"), "Problem not initialized"
        self.step += 1
        self.nt[self.step] = n
        self.gt[self.step] = n*self.dz
        Omtip = (2/3)*self.Kstar*self.dz**0.5
        Omk, domold = self.guess_opening(Omtip, n)
        
        zeta = self.get_coordinates(n)
        A, Mnn, Mns, Msn, Mss = create_DD_Matrix(zeta, n, self.gt[self.step])

        # FLUID FRONT GUESSING
        m0 = self.mft[self.step-1]
        V = self.Vt[self.step-1]
        phi0 = 0.0
        gf = n*self.dz
        mf = n
        # mf = limit_fluidjump(mf, m0, n, self.mjump)
        # Loop for fluid front
        mf, phi, V, dtnew = self.search_fluidfront(V, gf, mf, n, domold, A, Omk, Omtip)
        self.store_values(A, Omk, mf, phi, V, dtnew)
        if self.step % 1 == 0 :
            print("Step:" ,self.step, "Time", self.tau[self.step])
        
    def update_dz(self):
        g0 = self.gt[self.step]
        self.dz = g0/self.n0
        
    def remesh(self):
        print("Remeshing...")
        mfref = int(self.mft[self.step])
        self.update_dz()
        # La fraccion de fluido solido siempre es 1
        xf = self.gft[self.step]/self.gt[self.step]
        Omref = self.Om[:self.nmax, self.step]
        Pcref = self.Pr[:mfref, self.step]
        # m0 siempre va a ser self.n0 porque xf es 1
        m0 = floor(self.n0*xf)
        # En consecuencia el phi0 es 0 siempre porque xf es 1 self.n0 es m0
        phi0 = self.n0*xf - m0
        self.mft[self.step] = m0
        self.phit[self.step] = phi0
        
        #xi = self.get_coordinates_unit(self.n0)
        #xiref = self.get_coordinates_unit(self.nmax)
        #Omint = interp1d(xiref, Omref, kind="cubic")(m0/self.n0)
        #self.Om[:self.n0, self.step] = interp1d(xiref, Omref, kind="cubic")(xi)
        
        #Pc = interp1d(xiref[:mfref], Pcref, kind="cubic", fill_value="extrapolate")(xi[:m0])
        #Pm = Pc[m0-1]
        
        if False:
            if m0 > 1:
                Pmm1 = Pc[m0-2]
                a = (Pmm1 - Pm - (Pm+self.Sstar)/(0.5+phi0))/((1.5+phi0)*self.dz**2)
                b = -(Pmm1 - Pm)/self.dz - a*2*(m0-1)*self.dz
                gradPint = 2*a*m0*self.dz + b
                V01 = -Omint**2*gradPint

            else:
                raise
            
        #self.Vt[self.step] = V01
        self.xft[self.step] = xf
        self.dtold = self.dtold*self.nmax/self.n0
                
    def guess_opening(self, Omtip, n):
        domold = np.zeros((n, 1))
        Om0 = self.Om[:n, self.step-1].reshape((n, 1))

        if n > (self.n0+1):
            domold[1:n, 0] = self.Om[:n-1, self.step-1] - self.Om[:n-1, self.step-2]
            domold[0, 0] = domold[1, 0]
        Omk = Om0 + domold
        Omk[-1] = Omtip
        return Omk, domold
        
    def search_fluidfront(self, V, gf, mf, n, domold, A, Omk, Omtip, d=1):
        phi = 0.0
        dtold = self.dtold            
        invA_lag, inv_All_Alc, vecs0, D, S = construct_components_elasticity(n, mf, A, self.dz, self.Sstar)
            
        dtnew = self.search_timestep(dtold,mf, A, inv_All_Alc, vecs0, D, S, domold, phi, Omk, Omtip, n)
        # Buscamos la velocidad del frente de fluido
        # TODO aqui cancele el computo de la velocidad de fluido
        V = self.dz/dtnew

        return mf, phi, V, dtnew
            
    def search_timestep(self, dtold, mf, A, AllinvAlc, vecs0, D, S, domold, phi, Omk, Omtip, n):
        k2 = 0
        converged = False
        Omk[n-1, 0] = Omtip
        search_root = Secante(dtold, self.DTRATIO, tol=self.eps2)
        ts = list()
        Fs = list()
        while (k2 < self.k2max) and (not converged):
            Omk[n-1, 0] = Omtip
            Omk = self.search_opening(mf, Omk, A, AllinvAlc, vecs0, D, S, domold, dtold)
            F1 = Omk[n-1, 0] - Omtip
            #ts.append(self.tau[self.step] + dtold)
            ts.append(k2)
            Fs.append(F1)
            dtnew, converged = search_root.ite(F1)
            dtold = dtnew
            k2 += 1
        #plt.scatter(ts, Fs)
        #plt.show(block=True)
        del search_root
        return dtnew
    
    def search_opening(self, mf, Omk, A, AllinvAlc, vecs0, D, S, domold, dtold):
        ind = 1
        k1 = 0
        err1 = 1
        dtdz3 = dtold / self.dz**3
        Om0 = self.Om[:, self.step - 1, None]
        domnew = domold.copy()
        
        while (k1 < self.k1max) and (err1 > self.eps1):
            k1 += 1
            # Computo flujo al final del canal
            flux = 0.0
            tr = 0 # TODO Aplicar la correccion para la tasa de inyeccion
            Mat, rhs, prec = build_matrices(Om0, Omk, mf, dtold, D, S, self.dz, dtdz3, flux, tr)
            #prec = None
            res, flag = bicgstab(Mat, rhs, rtol=self.eps1*10**(-3), atol=self.eps1*10**(-3), M=prec, x0=domold[:mf], maxiter=100)
            domnew[:mf] = np.expand_dims(res, axis=1)

            Omk[:mf] = Om0[:mf] + domnew[:mf]
            # Check convergence
            err1 = sqrt(np.sum((domnew-domold)**2))/sqrt(np.sum(domnew**2))
            domold[:] = domnew[:]
        return Omk
    
    def compute_error(self, val, newval):
        if np.linalg.norm(val) > 0.0:
            err = abs(newval - val)/abs(val)
        else:
            err = np.linalg.norm(newval-val)/np.linalg.norm(newval)
        return err
        
    def store_values(self, A, Omk, mf, phi, V, dtnew):
        currStep = self.step
        n = self.nt[currStep]
        Pc = A[:mf, :n]@Omk[:n]/self.dz
        self.Om[:n, currStep, None] = Omk
        self.Pr[:mf, currStep, None] = Pc
        self.gft[currStep] = self.dz*(mf+phi)
        self.mft[currStep] = mf
        self.phit[currStep] = phi
        self.Vt[currStep] = V
        self.Vtipt[currStep] = self.dz/dtnew
        self.xft[currStep] = self.gft[currStep]/self.gt[currStep]
        self.tau[currStep] = self.tau[currStep - 1] + dtnew
        self.dtold = dtnew
        
    def get_coordinates(self, n):
        return self.dz*np.arange(0.5,n+0.5,step=1.0)

    def get_coordinates_unit(self, n):
        return np.arange(0.5,n+0.5,step=1.0)/n

if __name__ == "__main__":

    problem = FHProblem(data)
    max_steps = (problem.nmax - problem.n0)
    taumax = float(problem.taumax)
    for cycle in range(problem.ncycles):
        for i in range(max_steps):
            n = problem.n0 + i + 1
            problem.solve_step(n)
        if problem.tau[problem.step] > taumax:
            break
        problem.remesh()
    
    tau0 = float(problem.tau0)
    taumax = float(problem.taumax)
    if True:
        if False:
            plt.plot(problem.tau[:problem.step] ,problem.xft[:problem.step])
            plt.ylim(( np.min(problem.xft[:problem.step]),np.max(problem.xft[:problem.step])))
            plt.xlim((tau0, taumax))
            plt.title("Fluid fraction")
            plt.xscale("log")
            plt.show()

        Tdim, Ldim = dimensionalize()
        plt.plot(problem.tau[:problem.step] * Tdim, problem.gt[:problem.step] * Ldim)
        plt.title("Fluid front")
        plt.xscale("log")
        plt.yscale("log")
        plt.show()
        #plt.xlim((tau0*Tdim, taumax*Tdim))
        #plt.ylim((np.min(problem.gft[:problem.step]),np.max(problem.gft[:problem.step])))

        out = np.column_stack((problem.tau[:problem.step] * Tdim, problem.gt[:problem.step] * Ldim))
        rows = ["{},{}".format(i, j) for i, j in out] 
        text = "\n".join(rows)
        with open('outputs/out.csv', 'w') as f: 
            f.write(text)