import gc

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

    core = np.array(core, dtype=np.uint8)

    stepsize = int(sm.get_setting("VOXEL_MESH_STEP_SIZE"))

    core = np.pad(core, stepsize, 'constant', constant_values=0)

    verts, faces, normals, _ = measure.marching_cubes(core, step_size=stepsize, allow_degenerate=False)

    core_mesh = trimesh.Trimesh(verts, faces)

    return core_mesh


def model_all_cores(multiprocessing_pool=None, use_rois=True):
    print_notice("Converting cores to 3D objects...")
    cores = dm.get_cores_from_database()

    for core in [x[0] for x in cores]:
        core_stack = None
        base_dir = fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_ROI_MODELS if use_rois
                                        else fm.SpecialFolder.REAL_ASPHALT_3D_CORE_MODELS) + str(core) + '/' + str(core)

        aggregate_exists = fm.file_exists(base_dir + '_aggregate.stl')
        binder_exists = fm.file_exists(base_dir + '_binder.stl')
        void_exists = fm.file_exists(base_dir + '_void.stl')

        if not (aggregate_exists and binder_exists and void_exists):
            core_stack = ca.get_core_by_id(core, use_rois=use_rois, multiprocessing_pool=multiprocessing_pool)

        if not aggregate_exists:
            model_core("aggregate", np.array([x == 255 for x in core_stack], np.bool), core, use_rois=use_rois)

        if not binder_exists:
            model_core("binder", np.array([x == 127 for x in core_stack], np.bool), core, use_rois=use_rois)

        if not void_exists:
            model_core("void", np.array([x == 0 for x in core_stack], np.bool), core, use_rois=use_rois)

        gc.collect()  # TODO: Trimesh appears to have some form of memory leak/cache issue


def model_core(name, data_points, core_id, use_rois=True):
    print_notice("\tConverting " + name)

    core_mesh = voxels_to_mesh(data_points)

    base_dir = fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_ROI_MODELS if use_rois
                                    else fm.SpecialFolder.REAL_ASPHALT_3D_CORE_MODELS) + str(core_id) + '/'

    fm.create_if_not_exists(base_dir)

    model_dir = base_dir + str(core_id) + '_' + name + '.stl'
    save_mesh(core_mesh, model_dir)
