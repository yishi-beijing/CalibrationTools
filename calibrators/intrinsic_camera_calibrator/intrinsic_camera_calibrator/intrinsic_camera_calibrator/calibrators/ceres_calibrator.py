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


import logging
import threading
from typing import Dict
from typing import List

from ceres_intrinsic_camera_calibrator.ceres_intrinsic_camera_calibrator_py import calibrate
import cv2
from intrinsic_camera_calibrator.board_detections.board_detection import BoardDetection
from intrinsic_camera_calibrator.calibrators.calibrator import Calibrator
from intrinsic_camera_calibrator.camera_models.camera_model_factory import make_ceres_camera_model
from intrinsic_camera_calibrator.camera_models.ceres_camera_model import CeresCameraModel
from intrinsic_camera_calibrator.camera_models.ceres_camera_model import CeresCameraModelEnum
from intrinsic_camera_calibrator.parameter import Parameter
import numpy as np


class CeresCalibrator(Calibrator):
    def __init__(
        self, camera_model_type: CeresCameraModelEnum, lock: threading.RLock, cfg: Dict = {}
    ):
        super().__init__(camera_model_type, lock, cfg)
        self.set_parameters(**cfg)

    def _calibration_impl(self, detections: List[BoardDetection]) -> CeresCameraModel:
        """Implement the calibrator interface."""
        height = detections[0].get_image_height()
        width = detections[0].get_image_width()

        camera_model = make_ceres_camera_model(self.camera_model_type)
        camera_model.calibrate(
            height=height,
            width=width,
            object_points_list=[
                detection.get_flattened_object_points() for detection in detections
            ],
            image_points_list=[detection.get_flattened_image_points() for detection in detections],
        )

        return camera_model
