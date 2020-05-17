from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np


def surface_plot(x_3d, y_3d, z_3d):
    x_grid, y_grid = np.meshgrid(x_3d, y_3d)
    # Set up plot
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    surf = ax.plot_surface(x_grid, y_grid, z_3d, rstride=1, cstride=1, cmap=cm.viridis,
                         linewidth=0, antialiased=False, shade=False)
    ax.plot_wireframe(x_grid, y_grid, z_3d, facecolors='r', cstride=0)

    plt.show()
#
# y = np.array([tau]*x.size)
# ax.plot3D(xs=x, ys=y, zs=smile)
#
# mask = df.tau == tau
# fig2, ax2 = plt.subplots()
# ax2.plot(x,smile)
# ax2.scatter(df.M_std[mask], df.iv[mask])