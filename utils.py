import numpy as np
import matplotlib.pyplot as plt


arr = np.genfromtxt("outputs/out.csv", delimiter=",")

toffset = 40
radial_M = 0.001*arr[toffset: ,0]**(4/9)
radial_K = 0.001*arr[toffset: ,0]**(2/5)
planstrain_MoK = 0.0005*arr[toffset: ,0]**(2/3)


plt.plot(arr[:, 0], arr[: ,1])
plt.plot(arr[toffset:, 0], radial_M)
plt.plot(arr[toffset:, 0], radial_K)
plt.plot(arr[toffset:, 0], planstrain_MoK)

plt.yscale("log")
plt.xscale("log")

plt.show()