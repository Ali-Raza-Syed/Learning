# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 23:15:39 2019

@author: Syed_Ali_Raza
"""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib import cm



# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
sx, sy = 1, 1
Z = X*sx + Y*sy

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

surf = [ax.plot_surface(X, Y, Z)]

ax.set_title('3D Test')
ax.set_zlim(-100, 100)

def update_lines(i, surf, x) :
    global sx, sy
    sx = sx + 0.1
    sy = sy + 0.1
    
    Z = X*sx + Y*sy
    Z[Z < i] = 0
    surf[0].remove()
    surf[0] = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Creating the Animation object
animate = animation.FuncAnimation(fig, update_lines, 50, fargs=(surf, X))

#from matplotlib.animation import FFMpegWriter
#
#writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
#animate.save("movie.mp4", writer=writer)

plt.show()