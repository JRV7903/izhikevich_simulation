import numpy as np
import matplotlib.pyplot as plt

spikes = np.loadtxt("spikes.csv", delimiter=",")
voltages = np.loadtxt("voltages.csv", delimiter=",")

fig, axes = plt.subplots(5, sharex=True)

for i in range(4):
    axes[i].plot(voltages[:,0], voltages[:,i+1])
axes[4].scatter(spikes[:,0], spikes[:,1], s=2)

axes[0].set_ylabel("Regular V [mV]")
axes[1].set_ylabel("Fast V [mV]")
axes[2].set_ylabel("Chattering V [mV]")
axes[3].set_ylabel("Bursting V [mV]")
axes[4].set_xlabel("T [ms]")
axes[4].set_ylabel("Spike indices")
plt.show()