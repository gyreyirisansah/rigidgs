#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Modified for VR-RigidGS: Supports both legacy labeled point clouds
# and new identity-learning path (no point labeling required).
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, 
                 load_iteration=None, shuffle=False, resolution_scales=[1.0], 
                 is_pretrain=False):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.backgroundId = 0
        self.scene_name = getattr(args, 'scene_name', os.path.basename(args.source_path))

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.inpainted_cameras = {}
        self.test_inpainted_cameras = {}

        self.object_id_to_name = {}
        self.object_name_to_newid = {}
        self.num_objects = 0

        self.is_pretrain = is_pretrain

        # ================================================================
        # Load scene info
        # ================================================================
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            extra_kwargs = {}
            if hasattr(args, 'masks_dir') and args.masks_dir:
                extra_kwargs['masks_dir'] = args.masks_dir
            if hasattr(args, 'num_objects') and args.num_objects:
                extra_kwargs['num_objects'] = args.num_objects
            if hasattr(args, 'llffhold') and args.llffhold is not None:
                extra_kwargs['llffhold'] = args.llffhold

            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.depths, args.eval, 
                args.train_test_exp, 
                inpainted_folder=getattr(args, 'inpainted_images', ''),
                **extra_kwargs
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.depths, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        # ================================================================
        # Determine mode: labeled vs identity-learning
        # ================================================================
        has_labels = (scene_info.object_id_array is not None and 
                      scene_info.object_newid_to_name is not None)
        
        self.num_objects = getattr(scene_info, 'num_objects', 0)

        if has_labels:
            print(f"\n[Scene] LABELED mode: {len(scene_info.object_newid_to_name)} object classes found")
            print("  Object ID to Name mapping:")
            for obj_id, obj_name in scene_info.object_newid_to_name.items():
                print(f"    ID {obj_id}: {obj_name}")
            self.object_id_to_name = scene_info.object_newid_to_name
            self.object_name_to_newid = scene_info.object_name_to_newid or {}
        else:
            print(f"\n[Scene] IDENTITY LEARNING mode: {self.num_objects} foreground objects")
            print("  Object identities will be learned from 2D masks during training")
            self.object_id_to_name = {0: "background"}
            self.object_name_to_newid = {"background": 0}
            for i in range(1, self.num_objects + 1):
                name = f"object_{i:03d}"
                self.object_id_to_name[i] = name
                self.object_name_to_newid[name] = i

        # ================================================================
        # Copy input PLY and save camera JSON
        # ================================================================
        if not self.loaded_iter:
            if scene_info.ply_path and os.path.exists(scene_info.ply_path):
                with open(scene_info.ply_path, 'rb') as src_file, \
                     open(os.path.join(self.model_path, "input.ply"), 'wb') as dest_file:
                    dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # ================================================================
        # Load cameras
        # ================================================================
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args, 
                scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args, 
                scene_info.is_nerf_synthetic, True)

            if self.is_pretrain:
                continue

            if scene_info.inpainted_cameras:
                print("Loading Inpainted Cameras")
                self.inpainted_cameras[resolution_scale] = cameraList_from_camInfos(
                    scene_info.inpainted_cameras, resolution_scale, args, 
                    scene_info.is_nerf_synthetic, False, is_inpainted=True)
            else:
                self.inpainted_cameras[resolution_scale] = []

            if scene_info.test_inpainted_cameras:
                print("Loading Test Inpainted Cameras")
                self.test_inpainted_cameras[resolution_scale] = cameraList_from_camInfos(
                    scene_info.test_inpainted_cameras, resolution_scale, args, 
                    scene_info.is_nerf_synthetic, True, is_inpainted=True)
            else:
                self.test_inpainted_cameras[resolution_scale] = []

        # ================================================================
        # Initialize or load Gaussians
        # ================================================================
        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(self.model_path, "point_cloud",
                             "iteration_" + str(self.loaded_iter),
                             "point_cloud.ply"),
                args.train_test_exp
            )
        else:
            # Pass object_ids from labeled point cloud (if available)
            self.gaussians.create_from_pcd(
                scene_info.point_cloud, 
                scene_info.train_cameras, 
                self.cameras_extent,
                object_ids=scene_info.object_id_array
            )

        # ================================================================
        # Set background ID
        # ================================================================
        self.backgroundId = 0
        if has_labels and scene_info.object_newid_to_name:
            for obj_id, obj_name in scene_info.object_newid_to_name.items():
                if obj_name.lower() == "background":
                    self.backgroundId = int(obj_id)
                    break

    # ================================================================
    # Accessor methods
    # ================================================================

    def get_name_to_id_mapping(self, object_name):
        return self.object_name_to_newid.get(object_name, 0)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }
        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

        with open(os.path.join(self.model_path, "object_id_to_name.json"), "w") as f:
            json.dump(self.object_id_to_name, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getInpaintedCameras(self, scale=1.0):
        if self.is_pretrain:
            return []
        return self.inpainted_cameras.get(scale, [])

    def getTestInpaintedCameras(self, scale=1.0):
        if self.is_pretrain:
            return []
        return self.test_inpainted_cameras.get(scale, [])

    def getBackgroundObjectID(self):
        return self.backgroundId
