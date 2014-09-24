# test
# test2

import numpy as np
import matplotlib.pyplot as plt

a = [1,2]
b = [1,0,0]
c = [1,1]
g = [0]

omega = 0.01 * 2 * np.pi
n_list = np.arange(0,4095,1)
u_list = np.sin(omega*n_list)

i = np.array([0],[0])

for n in np.nditer(n_list):
    i_nxt = np.array([0,0])
    i_current = i[:,n]
    u = u_list[n]
    i_nxt[0] = i_current[0] -a[0] + b[0]*u - g[0]*i_current[1]
    i_nxt[1] = i_current[0]
    

plt.plot(n,u)
plt.show()