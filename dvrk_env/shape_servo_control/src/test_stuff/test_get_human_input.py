



import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2*np.pi*t)


fig = plt.figure()
plt.plot(t,s)
# plt.show()
plt.draw()
plt.pause(0.5) 
# input("<Hit Enter To Close>")
val = input("Enter your value: ")
print(val)
plt.close(fig)