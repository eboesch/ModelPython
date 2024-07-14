import numpy as np
import matplotlib.pyplot as plt
from topography import fetch_topo
from namelist import topomx, topowd, topotype, nx, dx


# Use the same setup as when plotting the topography in readsim.py 
dx = dx / 1000.0
topomx = topomx / 1000.0
topowd = topowd / 1000.0
topo = np.zeros(nx)
x = np.arange(nx, dtype="float32")
x0 = (nx - 1) / 2.0 + 1
x = (x + 1 - x0) * dx

# Create the desired topography
topo[1:-1] = fetch_topo(x, topowd, topomx, topotype)

# Create a plot of of only the topography shape, no axes needed.
plt.rcParams['figure.figsize'] = [8, 1.5]
plot = plt.plot(x, topo)
plt.setp(plot, color='k', linewidth=2.0)
plt.axis('off') 
plt.tight_layout() 
plt.show()