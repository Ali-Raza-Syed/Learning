# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:06:14 2019

@author: Syed_Ali_Raza
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
ax = fig.gca(projection='3d')

x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)

X, Y = np.meshgrid(x, y)

R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

def animate(i):
    

ani0 = animation.FuncAnimation(
    fig, animate, interval=1, blit=True, save_count=500)

from matplotlib.animation import FFMpegWriter

writer = FFMpegWriter(fps=60, metadata=dict(artist='Me'), bitrate=1800)
ani0.save("movie.mp4", writer=writer)

plt.show()