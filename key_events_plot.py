# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:46:51 2019

@author: Syed_Ali_Raza
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:45:16 2019

@author: Syed_Ali_Raza
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.zeros([2, 100])
x[0, :] = np.arange(0, 10, 0.1)
x[1, :] = np.arange(0, 10, 0.1)

y = np.zeros([2, 100])
y[0, :] = np.arange(0, 10, 0.1)
y[1, :] = -np.arange(0, 10, 0.1)



def press(event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        y[0, :] -= 1/10
        y[1, :] += 1/10
        fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', press)

line0, = ax.plot(x[0, :], y[0, :])
line1, = ax.plot(x[1, :], y[1, :])

plt.show()