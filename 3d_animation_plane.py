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
import matplotlib


# Make data.
X1 = np.arange(-20, 20, 0.25)
X2 = np.arange(-20, 20, 0.25)
X1, X2 = np.meshgrid(X1, X2)
w1, w2 = 100, 100
Z = w1*X1 + w2*X2

learning_rate = 0.001
p_x1 = 2
p_x2 = 3
p_z = 10
dw1 = 0
dw2 = 0

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

surf = [ax.plot_surface(X1, X2, Z, alpha = 0.5), ax.scatter([p_x1], [p_x2], p_z, c=[0], cmap=matplotlib.colors.ListedColormap(['red']))]

ax.set_title('3D Test')
ax.set_zlim(-100, 100)

def update_lines(i, surf, x) :
    global w1, w2, dw1, dw2
    
    dw1 = p_x1 * (2*p_x1*w1 + 2*p_x2*w2 - 2*p_z)
    dw2 = p_x2 * (2*p_x1*w1 + 2*p_x2*w2 - 2*p_z)
    
    w1 = w1 - learning_rate*dw1
    w2 = w2 - learning_rate*dw2
    
    Z = w1*X1 + w2*X2
    
    J = (p_x1*w1 + p_x2*w2 - p_z)**2
    print(J)
#    Z[Z < i] = 0
    surf[0].remove()
    surf[1].remove()
    surf[0] = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha = 0.5)
    surf[1] = ax.scatter([p_x1], [p_x2], p_z, c=[0], cmap=matplotlib.colors.ListedColormap(['black']))

# Creating the Animation object
animate = animation.FuncAnimation(fig, update_lines, 50, fargs=(surf, X1))

#from matplotlib.animation import FFMpegWriter
#
#writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
#animate.save("movie.mp4", writer=writer)

plt.show()