# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:37:18 2019

@author: Syed_Ali_Raza
"""

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
line0, = ax.plot(x[0, :], y[0, :])
line1, = ax.plot(x[1, :], y[1, :])

def init():  # only required for blitting to give a clean slate.
    line0.set_ydata([np.nan] * len(x[0, :]))
    line1.set_ydata([np.nan] * len(x[1, :]))
    return [line0, line1]

def animate(i):
    y[0, :] -= 1/10
    y[1, :] += 1/10
    line0.set_ydata(y[0, :])  # update the data.
    line1.set_ydata(y[1, :])
    return [line0, line1]

ani0 = animation.FuncAnimation(
    fig, animate, init_func=init, interval=1, blit=True, save_count=500)

from matplotlib.animation import FFMpegWriter

writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)
ani0.save("movie.mp4", writer=writer)

plt.show()