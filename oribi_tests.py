import yaml 
import os 
import numpy as np
import matplotlib.pyplot as plt 

from oribi import FHProblem
from oribi_results_reference import get_all_references_case

with open("solver_tests_data.yml", "r") as stream:
    try:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    except yaml.YAMLError as exc:
        print(exc)

KSTARS_REF = [0.75, 1, 1.25, 1.51, 1.8, 2.15, 1.3]
SSTARS_REF = [10, 10, 25, 45, 90, 5, 1]

errors_xft = list()
errors_gft = list()
errors_gt = list()

for i in range(1, 7):
    # .txt Obtenidos del codigo original para comparar
    gt_ref, gft_ref, xft_ref, tau_ref = get_all_references_case(i)
    
    data["parameters"]["Kstar"] = KSTARS_REF[i-1]
    data["parameters"]["Sstar"] = SSTARS_REF[i-1]
    data["dat_filepath"] = f"Oribi_Overtex{i}.dat"
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

    # Xft
    uh = np.squeeze(problem.xft[:problem.step])
    uref = xft_ref[:problem.step]
    err = np.linalg.norm(uh - uref) * 100/ np.linalg.norm(uref)
    errors_xft.append(err)

    # gt
    uh = np.squeeze(problem.gt[:problem.step])
    uref = gt_ref[:problem.step]
    err = np.linalg.norm(uh - uref) * 100/ np.linalg.norm(uref)
    errors_gt.append(err)

    # gtf
    uh = np.squeeze(problem.gft[:problem.step])
    uref = gft_ref[:problem.step]
    err = np.linalg.norm(uh - uref) * 100/ np.linalg.norm(uref)
    errors_gft.append(err)

print("ERRORES XFT", errors_xft)
print("ERRORES GFT", errors_gft)
print("ERRORES GT", errors_gt)