import gc
import sys

import numpy as np
import trimesh
from trimesh import caching
from numpy.linalg import norm
from multiprocessing import Process

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

    verts, faces, _, _ = measure.marching_cubes(core, step_size=stepsize, allow_degenerate=False)
    del _
    del core

    core_mesh = trimesh.Trimesh(verts, faces)
    print_notice("Verts size: %s" % str(sys.getsizeof(verts)), mt.MessagePrefix.DEBUG)
    print_notice("Faces size: %s" % str(sys.getsizeof(faces)), mt.MessagePrefix.DEBUG)

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

        processes = list()

        if not (aggregate_exists and binder_exists and void_exists):
            core_stack = ca.get_core_by_id(core, use_rois=use_rois, multiprocessing_pool=multiprocessing_pool)

        if not aggregate_exists:
            processes.append(Process(target=model_core,
                                     args=("aggregate", np.array([x == 255 for x in core_stack], np.bool),
                                           core, use_rois,)))

        if not binder_exists:
            processes.append(Process(target=model_core,
                                     args=("binder", np.array([x == 127 for x in core_stack], np.bool), core, use_rois,)
                                     ))

        if not void_exists:
            processes.append(Process(target=model_core,
                                     args=("void", np.array([x == 0 for x in core_stack], np.bool), core, use_rois,)
                                     ))

        for process in processes:
            process.start()
            process.join()


def fix_mesh(mesh):
    print_notice("Fixing mesh...")

    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.rezero()

    return mesh


def model_core(name, data_points, core_id, use_rois=True):
    print_notice("\tConverting " + name)

    core_mesh = voxels_to_mesh(data_points)
    core_mesh = fix_mesh(core_mesh)

    base_dir = fm.compile_directory(fm.SpecialFolder.REAL_ASPHALT_3D_ROI_MODELS if use_rois
                                    else fm.SpecialFolder.REAL_ASPHALT_3D_CORE_MODELS) + str(core_id) + '/'

    fm.create_if_not_exists(base_dir)

    model_dir = base_dir + str(core_id) + '_' + name + '.stl'
    save_mesh(core_mesh, model_dir)
    caching.Cache.clear(core_mesh)

