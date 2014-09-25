# test
# test2

import numpy as np
import matplotlib.pyplot as plt
import spec_ana as sa

import ds

a = [1,2]
b = [1,0,0]
c = [1,1]
g = [0]
nbit = 2
mid_rise = 0
fs = 1e6
signal_bin_num = 11
OSR = 32
plot = 1
ylow = -90
des = 'Output Spectrum'

length = 4096
omega = 2 * np.pi / length * signal_bin_num
n_list = np.arange(0,length,1)
u_list = np.sin(omega*n_list)
v_list = []

i = np.array([[0.0],[0.0]])

for n in range(length):
    i_nxt = np.array([0.0,0.0])
    i_current = i[:,n]
    u = u_list[n]
    
    q_input = b[2]*u + c[1]*i_current[1]
    q,qint = ds.quantize(q_input,2,0)
    v_list.append(q)
    
    i_nxt[0] = i_current[0] - a[0]*q + b[0]*u - g[0]*i_current[1]
    i_nxt[1] = i_current[1] - a[1]*q + b[1]*u + c[0]*i_current[0]
    i = np.column_stack((i,i_nxt))
    
v_list = np.array(v_list)
plt.plot(n_list,v_list)
plt.show()
sa.myfft(v_list,fs,signal_bin_num,OSR,plot,ylow,des)