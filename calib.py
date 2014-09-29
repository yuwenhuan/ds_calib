import numpy as np
import matplotlib.pyplot as plt

import spec_ana as sa
import ds

length = 1024*64
dac_mismatch = np.array([0.45,-0.55,0.95,-0.85])
u_list = 0.25 * (1 - 0.55*0.01) * np.ones(length)
a = [1,2]
b = [1,0,0]
c = [1,1]
g = [0]
q_bit = 1
q_type = 1
dwa = 0
dac_mismatch = np.array([(0.45+0.95-0.85)*0.01*0.25])

fs = 1e6
signal_bin_num = 23
OSR = 32
plot = 1
ylow = -160
des = 'Output Spectrum'

v_list = ds.sim_cifb_2(u_list,a,b,c,g,q_bit,q_type,dwa,dac_mismatch)
mean_v = np.mean(v_list)
print mean_v-0.25