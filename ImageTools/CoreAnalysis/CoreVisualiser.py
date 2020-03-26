import matplotlib.pyplot as plt
import numpy as np
import trimesh

from Settings import SettingsManager as sm, MessageTools as mt
from Settings.MessageTools import print_notice
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from stl import mesh

if sm.display_available:
    from mayavi import mlab


def plot_core(core):
    if not sm.display_available:
        mt.print_notice("No displays available - Cannot show 3D core visualisation!", mt.MessagePrefix.ERROR)

    pts = mlab.plot3d(core)


def voxels_to_mesh(core):
    print_notice("Converting image stack of voxels to 3D mesh... ", mt.MessagePrefix.INFORMATION, end='')

    aggregates = np.array([x == 255 for x in core], np.bool)
    # binders = [x == 1 for x in core]

    verts, faces, _, _ = measure.marching_cubes_lewiner(aggregates)

    core_mesh = trimesh.Trimesh(verts, faces)

    print("done")

    return core_mesh
