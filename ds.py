import numpy as np

def quantize(x, n,type=0):
    # x: input
    # n: number of bits
    # type: 0: 'mid-tread' /1: 'mid-rise'
    # return (q,q_int)

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

    return (q, q_int)