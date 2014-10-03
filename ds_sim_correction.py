# simulate delta-sigma modulator with digital correction

import numpy as np
import matplotlib.pyplot as plt

import spec_ana as sa
import ds

length = 8192*2
signal_bin_num = 23
omega = 2 * np.pi / length * signal_bin_num
n_list = np.arange(0,length,1)
u_list = 0.5*np.cos(omega*n_list)
a = [1,2]
b = [1,0,0]
c = [1,1]
g = [0]
q_bit = 2
q_type = 0
dwa = 0
dac_mismatch = np.array([0.45,-0.55,0.95,-0.85])
dac_corr = np.array([0.45003461804754841, -0.54878895938499606, 0.95040608259892767, -0.84834156178290776])

fs = 1e6
signal_bin_num = 23
OSR = 8
plot = 1
ylow = -160
des = 'Output Spectrum'

v_list,dac_used_list = ds.sim_cifb_2(u_list,a,b,c,g,q_bit,q_type,dwa,dac_mismatch)
dac_corr = np.dot(dac_used_list,dac_mismatch)/100/dac_mismatch.size
delay = 2
dac_corr = np.append(np.zeros(delay),dac_corr[:-delay])
v_list_corr = v_list + dac_corr
plt.plot(n_list,v_list)
plt.show()
V=sa.fft(v_list_corr,fs,signal_bin_num,OSR,plot,ylow,des)