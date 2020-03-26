import matplotlib.pyplot as plt
import numpy as np

from Settings import SettingsManager as sm, MessageTools as mt
from Settings.MessageTools import print_notice
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from skimage import measure

if sm.display_available:
    from mayavi import mlab


def plot_core(core):
    if not sm.display_available:
        mt.print_notice("No displays available - Cannot show 3D core visualisation!", mt.MessagePrefix.ERROR)

    pts = mlab.plot3d(core)


def voxels_to_mesh(core):
    print_notice("Converting image stack of voxels to 3D mesh... ", mt.MessagePrefix.INFORMATION, end='')

    core = np.array(core, dtype=np.uint8)

    verts, faces, normals, values = measure.marching_cubes_lewiner(core)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces])
    mesh.set_edgecolor('k')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis: a = 6 per ellipsoid")
    ax.set_ylabel("y-axis: b = 10")
    ax.set_zlabel("z-axis: c = 16")

    ax.set_xlim(0, 24)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, 20)  # b = 10
    ax.set_zlim(0, 32)  # c = 16

    plt.tight_layout()
    plt.show()

    print("done")
