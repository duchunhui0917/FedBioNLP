import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,10,1000)
y = np.sin(x)
a = list(range(0,11,2))
plt.plot(x,y)

plt.xticks(a)

plt.show()
