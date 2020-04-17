import numpy as np
import pymesh
from numpy.linalg import norm

from ImageTools.CoreAnalysis import CoreAnalyser as ca
from Settings import SettingsManager as sm, MessageTools as mt, DatabaseManager as dm, FileManager as fm
from Settings.MessageTools import print_notice
from skimage import measure


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
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)

    print_notice("\tSplitting long edges... ", mt.MessagePrefix.INFORMATION)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)

    num_vertices = mesh.num_vertices
    while True:
        print_notice("\tCollapsing short edges... ", mt.MessagePrefix.INFORMATION)
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len, preserve_feature=True)

        print_notice("\tRemoving obtuse triangles... ", mt.MessagePrefix.INFORMATION)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
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

    print_notice("\tVertices before: %d Faces before: %d\t" % (mesh.num_vertices, mesh.num_faces), mt.MessagePrefix.DEBUG)
    mesh = fix_mesh(mesh, "high")
    print_notice("\tVertices after: %d Faces after: %d\t" % (mesh.num_vertices, mesh.num_faces), mt.MessagePrefix.DEBUG)

    return mesh


def tetrahedralise_mesh(mesh):
    return pymesh.tetrahedralize(mesh, 1)


def save_mesh(mesh, directory):
    pymesh.save_mesh(directory, mesh)


def voxels_to_mesh(core, suppress_messages=False):
    if not suppress_messages:
        print_notice("Converting image stack of voxels to 3D mesh... ", mt.MessagePrefix.INFORMATION)

    core = np.array(core, dtype=np.float)

    stepsize = int(sm.configuration.get("VOXEL_MESH_STEP_SIZE"))

    core = np.pad(core, stepsize, 'constant', constant_values=0)

    verts, faces, normals, _ = measure.marching_cubes(core, step_size=stepsize, allow_degenerate=False)

    core_mesh = pymesh.form_mesh(verts, faces)

    return core_mesh


def model_all_cores():
    print_notice("Converting cores to 3D objects...")
    cores = dm.get_cores_from_database()

    for core in [x[0] for x in cores]:
        core_stack = None

        segment = []
        model = []

        if not fm.file_exists(fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) +
                          str(core) + '_aggregate.stl'):
            if core_stack is None:
                core_stack = ca.get_core_by_id(core)
            aggregate = np.array([x == 255 for x in core_stack], np.bool)
            segment += ["aggregate"]
            model += [aggregate]

        if not fm.file_exists(fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) +
                              str(core) + '_binder.stl'):
            if core_stack is None:
                core_stack = ca.get_core_by_id(core)
            binder = np.array([x == 127 for x in core_stack], np.bool)
            segment += ["binder"]
            model += [binder]

        if not fm.file_exists(fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) +
                          str(core) + '_void.stl'):
            if core_stack is None:
                core_stack = ca.get_core_by_id(core)
            void = np.array([x == 0 for x in core_stack], np.bool)
            segment += ["void"]
            model += [void]

        for ind in range(len(model)):
            print_notice("\tConverting " + segment[ind])
            core_mesh = voxels_to_mesh(model[ind])
            # core_mesh = cv.simplify_mesh(core_mesh)
            #core_mesh = cv.tetrahedralise_mesh(core_mesh)
            model_dir = fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) + str(core) + '_' + segment[ind] + '.stl'
            save_mesh(core_mesh, model_dir)