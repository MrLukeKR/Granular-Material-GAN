import numpy as np
import trimesh
from numpy.linalg import norm

from ImageTools.CoreAnalysis import CoreAnalyser as ca
from Settings import SettingsManager as sm, MessageTools as mt, DatabaseManager as dm, FileManager as fm
from Settings.MessageTools import print_notice
from skimage import measure


def save_mesh(mesh, directory):
    mesh.export(directory)


def voxels_to_mesh(core, suppress_messages=False):
    if not suppress_messages:
        print_notice("Converting image stack of voxels to 3D mesh... ", mt.MessagePrefix.INFORMATION)

    core = np.array(core, dtype=np.float)

    stepsize = int(sm.configuration.get("VOXEL_MESH_STEP_SIZE"))

    core = np.pad(core, stepsize, 'constant', constant_values=0)

    verts, faces, normals, _ = measure.marching_cubes(core, step_size=stepsize, allow_degenerate=False)

    core_mesh = trimesh.Trimesh(verts, faces)

    return core_mesh


def model_all_cores():
    print_notice("Converting cores to 3D objects...")
    cores = dm.get_cores_from_database()

    for core in [x[0] for x in cores]:
        core_stack = None

        if not fm.file_exists(fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) +
                          str(core) + '_aggregate.stl'):
            if core_stack is None:
                core_stack = ca.get_core_by_id(core)
            model_core("aggregate", np.array([x == 255 for x in core_stack], np.bool), core)

        if not fm.file_exists(fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) +
                              str(core) + '_binder.stl'):
            if core_stack is None:
                core_stack = ca.get_core_by_id(core)
            model_core("binder", np.array([x == 127 for x in core_stack], np.bool), core)

        if not fm.file_exists(fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) +
                          str(core) + '_void.stl'):
            if core_stack is None:
                core_stack = ca.get_core_by_id(core)
            model_core("void", np.array([x == 0 for x in core_stack], np.bool), core)


def model_core(name, datapoints, core_id):
    print_notice("\tConverting " + name)
    core_mesh = voxels_to_mesh(datapoints)
    model_dir = fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_MODELS) + str(core_id) + '_' + name + '.stl'
    save_mesh(core_mesh, model_dir)
