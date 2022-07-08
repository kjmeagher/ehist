import numpy as np
import pylab as plt

from ehist.ansi_cmap import ansi_cmap

x1 = np.linspace(0, 4, 60)
y1 = np.linspace(0, 10, 50)

xx, yy = np.meshgrid(x1, y1)
z1 = 30 * xx * np.exp(-xx) - yy
print(ansi_cmap(x1, y1, z1))

import pylab as plt

plt.pcolormesh(xx, yy, z1, shading="auto", cmap=None)
plt.colorbar()
plt.show()
