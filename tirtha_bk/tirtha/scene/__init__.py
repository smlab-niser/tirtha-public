"""
The contents of this file are subject to the terms of the Gaussian-Splatting LICENSE present in ./LICENSE.md.

"""
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import random
from pathlib import Path

# Local imports
from .dataset_readers import sceneLoadTypeCallbacks
from .gaussian_model import GaussianModel
from ..utility.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    def __init__(self, ModelParams, gaussians:GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = ModelParams.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                point_cloud_path = Path(str(self.model_path) / "point_cloud")
                point_cloud_path.mkdir(parents=True, exist_ok=True)
                self.loaded_iter = searchForMaxIteration(point_cloud_path)
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        
        if Path(str(ModelParams.source_path) / "sparse").exists():
            scene_info = sceneLoadTypeCallbacks["Colmap"](ModelParams.source_path, ModelParams.images, ModelParams.eval)
        elif Path(str(ModelParams.source_path) / "transforms_train.json").exists():
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](ModelParams.source_path, ModelParams.white_background, ModelParams.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(Path(str(self.model_path) / "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(Path(str(self.model_path) / "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, ModelParams)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, ModelParams)

        if self.loaded_iter:
            point_cloud_path_final = Path(str(self.model_path) /
                                                           "point_cloud" /
                                                           "iteration_"  / str(self.loaded_iter) /
                                                           "point_cloud.ply")
            point_cloud_path_final.mkdir(parents=True, exist_ok=True)
            self.gaussians.load_ply(point_cloud_path_final)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = Path(str(self.model_path) / "point_cloud/iteration_{}".format(iteration) / "point_cloud.ply")
        self.gaussians.save_ply(point_cloud_path)


    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
def searchForMaxIteration(folder):
    folder = list(Path(folder).iterdir())
    saved_iters = [int(fname.split("_")[-1]) for fname in folder]
    return max(saved_iters)
