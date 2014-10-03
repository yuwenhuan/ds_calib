import numpy as np
import matplotlib.pyplot as plt

import spec_ana as sa
import ds

length = 512
dac_mismatch = np.array([0.45,-0.55,0.95,-0.85])
#dac_mismatch = np.array([0,0,0,0])

n_dac = dac_mismatch.size
dac_corr = []
for calib_element in range(n_dac):
    dac_active = np.ones(n_dac)
    dac_active[calib_element] = 0
    dac_total = 1.0/n_dac * np.dot((1+dac_mismatch/100),dac_active)
    dac_nominal = (n_dac-1.0)/n_dac
    u = (1+dac_mismatch[calib_element]/100)
    u_list = u * np.ones(length)
    a = [1,2]
    b = [1.0/(n_dac-1),0,0]
    c = [1,1]
    g = [0]
    q_bit = 1
    q_type = 1
    dwa = 0
    dac_mismatch_1bit = np.array([100*(dac_total/dac_nominal-1)])

    
    v_list,dac_used_list = ds.sim_cifb_2(u_list,a,b,c,g,q_bit,q_type,dwa,dac_mismatch_1bit)
    d = ds.sum2(v_list[2:])
    #d = np.mean(v_list[2:])
    e = ((n_dac-1)*d-1)/(1+d)*100
    print e
    print dac_mismatch[calib_element]
    dac_corr.append(e)
print dac_corr