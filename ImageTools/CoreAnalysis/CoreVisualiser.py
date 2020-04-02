import numpy as np
import pymesh
import trimesh
from numpy.linalg import norm

from Settings import SettingsManager as sm, MessageTools as mt
from Settings.MessageTools import print_notice
from skimage import measure

if sm.display_available:
    from mayavi import mlab


def plot_core(core):
    if not sm.display_available:
        mt.print_notice("No displays available - Cannot show 3D core visualisation!", mt.MessagePrefix.ERROR)

    pts = mlab.plot3d(core)


def fix_mesh(mesh, detail="normal"):
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)

    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 1e-2
    else:
        target_len = None

    count = 0
    print_notice("\tRemoving degenerated triangles... ", mt.MessagePrefix.INFORMATION)
    mesh, __ = pymesh.remove_degenerated_triangles(mesh)

    print_notice("\tSplitting long edges... ", mt.MessagePrefix.INFORMATION)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)

    num_vertices = mesh.num_vertices
    while True:
        print_notice("\tCollapsing short edges... ", mt.MessagePrefix.INFORMATION)
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)

        print_notice("\tRemoving obtuse triangles... ", mt.MessagePrefix.INFORMATION)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print_notice("\tVertices: %d Faces: %d\t" % (mesh.num_vertices, mesh.num_faces), mt.MessagePrefix.DEBUG)
        count += 1
        if count > 10:
            break

    print_notice("\tResolving self intersection... ", mt.MessagePrefix.INFORMATION)
    mesh = pymesh.resolve_self_intersection(mesh)

    print_notice("\tRemoving duplicated faces... ", mt.MessagePrefix.INFORMATION)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)

    print_notice("\tComputing outer hull... ", mt.MessagePrefix.INFORMATION)
    mesh = pymesh.compute_outer_hull(mesh)

    print_notice("\tRemoving duplicated faces... ", mt.MessagePrefix.INFORMATION)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)

    print_notice("\tRemoving obtuse triangles... ", mt.MessagePrefix.INFORMATION)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)

    print_notice("\tRemoving isolated vertices... ", mt.MessagePrefix.INFORMATION)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh


def simplify_mesh(mesh):
    print_notice("Simplifying mesh to reduce file size... ", mt.MessagePrefix.INFORMATION)

    mesh = pymesh.form_mesh(mesh.vertices, mesh.faces)

    print_notice("\tVertices before: %d Faces before: %d\t" % (mesh.num_vertices, mesh.num_faces), mt.MessagePrefix.DEBUG)
    mesh = fix_mesh(mesh)
    print_notice("\tVertices after: %d Faces after: %d\t" % (mesh.num_vertices, mesh.num_faces), mt.MessagePrefix.DEBUG)

    mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

    return mesh


def voxels_to_mesh(core, suppress_messages=False):
    if not suppress_messages:
        print_notice("Converting image stack of voxels to 3D mesh... ", mt.MessagePrefix.INFORMATION, end='')

    core = np.array(core, np.uint8)

    verts, faces, _, _ = measure.marching_cubes(core)

    core_mesh = trimesh.Trimesh(verts, faces)
    # core_mesh = pymesh.form_mesh(verts, faces)

    if not suppress_messages:
        print("done")

    return core_mesh
