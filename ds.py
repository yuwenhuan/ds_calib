# this module is for delta-sigma modulator simulation with DAC mismatch

import numpy as np

def quantize(x, n,type=0):
    """
    x: input (from -1 to 1)
    n: number of bits
    type: 0: 'mid-tread' /1: 'mid-rise'
    
    return (q,q_int)
    q: quantized output (from -1 to 1)
    q_int: integer quantized output (from 0 to n_level-1)
    """

    if type == 0:
        q = np.round(x*np.power(2,n-1))/np.power(2,n-1)
        q_max = 1
        q_min = -1
    else:
        q = (np.round(x*np.power(2,n-1)+0.5)-0.5)/np.power(2,n-1)
        q_max = 1 - 0.5/np.power(2,n-1)
        q_min = -(1 - 0.5/np.power(2,n-1))
    
    if q > q_max:
        q = q_max
    elif q < q_min:
        q = q_min
    
    if type == 0:
        q_int = (q+1)*np.power(2,n-1)
    else:
        q_int = (q+1-1/(np.power(2,n)))*np.power(2,n-1)

    q_int = q_int.astype(np.int64)
    return (q, q_int)




def sim_cifb_2(u_list,a,b,c,g,q_bit,q_type,dwa,dac_mismatch):
    """
    function to simulate second order CIFB modulators
    
    Input:
    u_list: modulator input (a numpy array)
    a,b,c,g: delta-sigma coefficeints (Schreier and Temes book pp.412). 
    q_bit: number of bits for quantizer
    q_type: 0: 'mid-tread' /1: 'mid-rise'
    dwa: True/False
    dac_mismatch: numpy array given in percentage (average dac element as benchmark)
    
    Output:
    v_list: modulator output (numpy array)
    
    For example:
    u_list = 0.5*np.cos(omega*n_list)
    a = [1,2]
    b = [1,0,0]
    c = [1,1]
    g = [0]
    q_bit = 2
    q_type = 0
    dwa = 1
    dac_mismatch = np.array([0.45,-0.55,0.95,-0.85]) # for 0.45%, -0.55% etc.
    """
    
    length = u_list.size
    v_list = []
    dac_ptr = 0
    
    i = np.array([[0.0],[0.0]]) # integrator content

    for n in range(length):
        i_nxt = np.array([0.0,0.0])
        i_current = i[:,n]
        u = u_list[n]
        
        q_input = b[2]*u + c[1]*i_current[1]
        q,qint = quantize(q_input,2,0)
        v_list.append(q)
        
        # DAC
        if q_type == 0:
            n_dac = np.power(2,q_bit)
        else:
            n_dac = np.power(2,q_bit) - 1
        if dwa == 0:
            dac_used = np.append(np.ones(qint),np.zeros(n_dac-qint))
        else:
            dac_used = np.zeros(n_dac)
            for ii in range(qint):
                dac_used[dac_ptr] = 1
                dac_ptr = (dac_ptr + 1) % n_dac
        dac_error = 1.0/4*0.01*np.dot(dac_used,dac_mismatch)
        dac_out = q + dac_error
        
        i_nxt[0] = i_current[0] - a[0]*dac_out + b[0]*u - g[0]*i_current[1]
        i_nxt[1] = i_current[1] - a[1]*dac_out + b[1]*u + c[0]*i_current[0]
        i = np.column_stack((i,i_nxt))
        
    return np.array(v_list)

