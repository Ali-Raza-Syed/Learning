# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 19:45:16 2019

@author: Syed_Ali_Raza
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

x = np.arange(0, 10, 0.1)
p_x = 2
p_y = 5
w = -10
l_r = 0.001
epochs = 1000
dJ = 0

fig, ax = plt.subplots()

x = np.arange(0, 10, 0.1)
line, = ax.plot(x, x)

def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(x))
    return line,


def animate(i):
    global w
    y_line = w * x
    dJ = 4 * p_x * w - 4 * p_y
    w = w - l_r * dJ
    
    y_data = y_line
    y_data[21] = p_y
    line.set_ydata(y_data)  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=1, blit=True, save_count=500)

from matplotlib.animation import FFMpegWriter

writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)
ani.save("movie.mp4", writer=writer)

plt.show()