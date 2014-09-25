import ds
reload(ds)
import numpy as np
import matplotlib.pyplot as plt

x_a = np.arange(-1,1,0.01)
y_a = np.zeros(x_a.size)
for ii in range(x_a.size):
    q,qint = ds.quantize(x_a[ii],2,0)
    y_a[ii] = q

plt.plot(x_a,y_a)
plt.show()