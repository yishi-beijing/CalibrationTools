#!/usr/bin/env python3

# Copyright 2024 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import threading
from typing import Dict
from typing import List

import cv2
from intrinsic_camera_calibrator.board_detections.board_detection import BoardDetection
from intrinsic_camera_calibrator.calibrators.calibrator import Calibrator
from intrinsic_camera_calibrator.camera_models.opencv_camera_model import OpenCVCameraModel
from intrinsic_camera_calibrator.camera_models.opencv_camera_model import OpenCVCameraModelEnum
from intrinsic_camera_calibrator.camera_models.camera_model_factory import make_opencv_camera_model
from intrinsic_camera_calibrator.parameter import Parameter


class OpenCVCalibrator(Calibrator):
    """Wrapper of the opencv's camera calibration routine."""

    def __init__(self, camera_model_type: OpenCVCameraModelEnum, lock: threading.RLock, cfg: Dict = {}):
        super().__init__(camera_model_type, lock, cfg)
        # self.radial_distortion_coefficients = Parameter(int, value=2, min_value=0, max_value=6)
        # self.use_tangential_distortion = Parameter(
        #     bool, value=True, min_value=False, max_value=True
        # )
        # self.fix_principal_point = Parameter(bool, value=False, min_value=False, max_value=True)
        # self.fix_aspect_ratio = Parameter(bool, value=False, min_value=False, max_value=True)

        self.set_parameters(**cfg)

    def _calibration_impl(self, detections: List[BoardDetection]) -> OpenCVCameraModel:
        """Implement the calibrator interface."""
        height = detections[0].get_image_height()
        width = detections[0].get_image_width()

        camera_model = make_opencv_camera_model(self.camera_model_type)
        camera_model.calibrate(
            height=height,
            width=width,
            object_points_list=[
                detection.get_flattened_object_points() for detection in detections
            ],
            image_points_list=[detection.get_flattened_image_points() for detection in detections]
        )

        return camera_model
