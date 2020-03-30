import numpy as np
# import pymesh
import trimesh

from Settings import SettingsManager as sm, MessageTools as mt
from Settings.MessageTools import print_notice
from skimage import measure

if sm.display_available:
    from mayavi import mlab


def plot_core(core):
    if not sm.display_available:
        mt.print_notice("No displays available - Cannot show 3D core visualisation!", mt.MessagePrefix.ERROR)

    pts = mlab.plot3d(core)


def simplify_mesh(mesh):
    # TODO: Decimate/Simplify mesh
    print_notice("Simplifying mesh to reduce file size... ", mt.MessagePrefix.INFORMATION, end='')
    print("done")
    return mesh


def voxels_to_mesh(core, suppress_messages=False):
    if not suppress_messages:
        print_notice("Converting image stack of voxels to 3D mesh... ", mt.MessagePrefix.INFORMATION, end='')

    core = np.array(core, np.uint8)

    verts, faces, _, _ = measure.marching_cubes_lewiner(core)

    core_mesh = trimesh.Trimesh(verts, faces)
    # core_mesh = pymesh.form_mesh(verts, faces)

    if not suppress_messages:
        print("done")

    return core_mesh
