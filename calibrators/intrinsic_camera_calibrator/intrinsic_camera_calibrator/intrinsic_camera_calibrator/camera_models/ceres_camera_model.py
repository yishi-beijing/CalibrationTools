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

from typing import List
from typing import Optional

import cv2
import numpy as np

from ceres_intrinsic_camera_calibrator.ceres_intrinsic_camera_calibrator_py import calibrate
from intrinsic_camera_calibrator.camera_models.camera_model import CameraModel
from intrinsic_camera_calibrator.camera_models.camera_model import CameraModelEnum
from intrinsic_camera_calibrator.camera_models.opencv_camera_model import PolynomialOpenCVCameraModel
from intrinsic_camera_calibrator.camera_models.opencv_camera_model import RationalOpenCVCameraModel
from intrinsic_camera_calibrator.parameter import Parameter


class CeresCameraModelEnum(CameraModelEnum):
    CERES_POLYNOMIAL = {"name": "ceres_polynomial",
                        "display": "Ceres Polynomial", "calibrator": "ceres"}
    CERES_RATIONAL = {"name": "ceres_rational", "display": "Ceres Rational", "calibrator": "ceres"}


class CeresCameraModel(CameraModel):
    """Basic Ceres camera model class."""

    def __init__(
        self,
        k: Optional[np.array] = None,
        d: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ):
        super().__init__(k, d, height, width)

    def _init_calibrate_impl(self, **kwargs):
        """Abstract method to initial calibration of the Ceres camera model."""
        raise NotImplementedError

    def _calibrate_impl(
        self,
        object_points_list: List[np.array],
        image_points_list: List[np.array]
    ):
        """Calibrate Ceres camera model."""
        camera_model = self._init_calibrate_impl(object_points_list, image_points_list)

        rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate(
            object_points_list=object_points_list,
            image_points_list=image_points_list,
            initial_camera_matrix=camera_model.k,
            initial_dist_coeffs=camera_model.d,
            num_radial_coeffs=self.radial_distortion_coefficients,
            num_rational_coeffs=self.rational_distortion_coefficients,
            use_tangential_distortion=self.use_tangential_distortion,
            verbose=False
        )

        self.k = camera_matrix
        self.d = dist_coeffs

    def _get_undistorted_camera_model_impl(self, alpha: float):
        """Compute the undistorted version of the camera model."""
        undistorted_k, _ = cv2.getOptimalNewCameraMatrix(
            self.k, self.d, (self.width, self.height), alpha
        )

        return CameraModel(
            k=undistorted_k, d=np.zeros_like(self.d), height=self.height, width=self.width
        )

    def _rectify_impl(self, img: np.array, alpha=0.0) -> np.array:
        """Rectifies an image using the current camera model. Alpha is a value in the [0,1] range to regulate how the rectified image is cropped. 0 means that all the pixels in the rectified image are valid whereas 1 keeps all the original pixels from the unrectified image into the rectifies one, filling with zeroes the invalid pixels."""
        if np.abs(self.d).sum() == 0:
            return img

        if self._cached_undistorted_model is None or alpha != self._cached_undistortion_alpha:
            self._cached_undistortion_alpha = alpha
            self._cached_undistorted_model = self.get_undistorted_camera_model(alpha=alpha)
            (
                self._cached_undistortion_map_x,
                self._cached_undistortion_map_y,
            ) = cv2.initUndistortRectifyMap(
                self.k, self.d, None, self._cached_undistorted_model.k, (self.width, self.height), 5
            )

        return cv2.remap(
            img, self._cached_undistortion_map_x, self._cached_undistortion_map_y, cv2.INTER_LINEAR
        )


class PolynomialCeresCameraModel(CeresCameraModel):
    """Polynomial Ceres camera model class."""

    def __init__(
        self,
        k: Optional[np.array] = None,
        d: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ):
        super().__init__(k, d, height, width)
        self.radial_distortion_coefficients = 3
        self.rational_distortion_coefficients = 0
        self.use_tangential_distortion = False

    def _init_calibrate_impl(
        self,
        object_points_list: List[np.array],
        image_points_list: List[np.array]
    ) -> PolynomialOpenCVCameraModel:
        """Initialize the calibration of the camera model."""
        camera_model = PolynomialOpenCVCameraModel()
        camera_model.calibrate(
            height=self.height,
            width=self.width,
            object_points_list=object_points_list,
            image_points_list=image_points_list
        )

        if (camera_model.d.size > 5):
            camera_model.d = camera_model.d.reshape(-1, camera_model.d.size)[:, :5]

        return camera_model


class RationalCeresCameraModel(CeresCameraModel):
    """Polynomial Ceres camera model class."""

    def __init__(
        self,
        k: Optional[np.array] = None,
        d: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None
    ):
        super().__init__(k, d, height, width)
        self.radial_distortion_coefficients = 3
        self.rational_distortion_coefficients = 3
        self.use_tangential_distortion = True

    def _init_calibrate_impl(
        self,
        object_points_list: List[np.array],
        image_points_list: List[np.array]
    ) -> RationalOpenCVCameraModel:
        """Initialize the calibration of the camera model."""
        camera_model = RationalOpenCVCameraModel()
        camera_model.calibrate(
            height=self.height,
            width=self.width,
            object_points_list=object_points_list,
            image_points_list=image_points_list
        )

        if (camera_model.d.size > 8):
            camera_model.d = camera_model.d.reshape(-1, camera_model.d.size)[:, :8]

        return camera_model
