# try to calibrate with DC offset

import numpy as np
import matplotlib.pyplot as plt

import spec_ana as sa
import ds

length = 200
dac_mismatch = np.array([0.45,-0.55,0.95,-0.85])
#dac_mismatch = np.array([0,0,0,0])
dc_offset = 0.01
seq = 2*(np.random.randint(0,2,length)-0.5)

calib_element = 2 # number of the dac element under calibration
n_dac = dac_mismatch.size
dac_active = np.ones(n_dac)
dac_active[calib_element] = 0
dac_total = 1.0/n_dac * np.dot((1+dac_mismatch/100),dac_active)
dac_nominal = (n_dac-1.0)/n_dac
u = 1+dac_mismatch[calib_element]/100
u_list = u * seq + dc_offset
a = [1,2]
b = [1.0/(n_dac-1),0,0]
c = [1,1]
g = [0]
q_bit = 1
q_type = 1
dwa = 0
dac_mismatch_1bit = np.array([100*(dac_total/dac_nominal-1)])

fs = 1e6
signal_bin_num = 23
OSR = 32
plot = 1
ylow = -160
des = 'Output Spectrum'

v_list = ds.sim_cifb_2(u_list,a,b,c,g,q_bit,q_type,dwa,dac_mismatch_1bit)
v_list_proc = v_list[2:] * seq[:-2]
#d = ds.sum2(v_list_proc)
d = np.mean(v_list_proc)
e1 = ((n_dac-1)*d-1)/(1+d)
print e1
print dac_mismatch[calib_element]