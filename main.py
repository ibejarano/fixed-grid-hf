from fracture_lag import Fracture, load_fracture_results
from meshing import CartesianMesh
from fracture_initializer import AxisymInitializer
from properties_conf import SimulationProperties
from oribi_results_reference import get_all_references_case, get_reference_adimensionals
import matplotlib.pyplot as plt
import numpy as np
###
ref_case = 1

Kref, Sref = get_reference_adimensionals(ref_case)
gt_ref, gft_ref, xft_ref, tau_ref = get_all_references_case(ref_case)
###

mesh = CartesianMesh()
initializer = AxisymInitializer(tau0=1e-16, name="Oribi_Overtex1.dat")
simProps = SimulationProperties()
simProps.set_simulation_name("test")
fracture = Fracture(mesh=mesh,initializer=initializer, simProps=simProps, Kstar=Kref, Sstar=Sref)
fracture.run()

results = load_fracture_results(name="test")
# Crear directorios
tmin, tmax = np.min(results[:, 1]), np.max(results[:, 1])
gmin, gmax = np.min(results[:, 2]), np.max(results[:, 2])
plt.plot(results[:, 1], results[:, 2])
plt.plot(tau_ref, gt_ref, "ro")
plt.xlim((tmin, tmax))
plt.ylim((gmin, gmax))
plt.show()