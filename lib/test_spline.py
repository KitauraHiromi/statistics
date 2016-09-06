import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import filter

x = np.array([-2,-1,1,2],dtype='float32')
y = np.array([4,1,5,1],  dtype='float32')
#plt.plot(x, y)
#plt.show()


xn = np.arange(-2.0, 2.1, 0.1)
rp = scipy.interpolate.splrep(x, y, s=0)
yn = scipy.interpolate.splev(xn, rp, der=0)
#plt.plot(x, y)
#plt.plot(xn, yn)
#plt.show()


line_x = np.arange(-2.0, 2.1, 0.7)
line_y = scipy.interpolate.splev(line_x, rp, der=0)
line_rp = filter.liner_interpolate(line_x, line_y, -2.0, 2.1, show=False)
plt.plot(x, y)
plt.plot(xn, yn)
plt.plot(line_x, line_y)
plt.show()
