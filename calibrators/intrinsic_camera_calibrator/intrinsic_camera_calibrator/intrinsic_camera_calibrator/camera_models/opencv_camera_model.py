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
from intrinsic_camera_calibrator.camera_models.camera_model import CameraModel
from intrinsic_camera_calibrator.camera_models.camera_model import CameraModelEnum
from intrinsic_camera_calibrator.parameter import Parameter
import numpy as np


class OpenCVCameraModelEnum(CameraModelEnum):
    OPENCV_POLYNOMIAL = {
        "name": "opencv_polynomial",
        "display": "OpenCV Polynomial",
        "calibrator": "opencv",
    }
    OPENCV_RATIONAL = {
        "name": "opencv_rational",
        "display": "OpenCV Rational",
        "calibrator": "opencv",
    }
    OPENCV_PRISM = {"name": "opencv_prism", "display": "OpenCV Prism", "calibrator": "opencv"}


class OpenCVCameraModel(CameraModel):
    """Basic opencv's camera model class."""

    def __init__(
        self,
        k: Optional[np.array] = None,
        d: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        super().__init__(k, d, height, width)
        self.flags = 0
        # self.flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        # self.flags |= cv2.CALIB_FIX_ASPECT_RATIO
        # self.flags |= cv2.CALIB_ZERO_TANGENT_DIST

    def _calibrate_impl(
        self, object_points_list: List[np.array], image_points_list: List[np.array]
    ):
        """Calibrate OpenCV camera model."""
        object_points_list = [
            object_points.astype(np.float32).reshape(-1, 3) for object_points in object_points_list
        ]
        image_points_list = [
            image_points.astype(np.float32).reshape(-1, 1, 2) for image_points in image_points_list
        ]

        _, self.k, self.d, rvecs, tvecs = cv2.calibrateCamera(
            object_points_list,
            image_points_list,
            (self.width, self.height),
            cameraMatrix=None,
            distCoeffs=None,
            flags=self.flags,
        )
        pass

    def _get_undistorted_camera_model_impl(self, alpha: float):
        """Compute the undistorted version of the camera model."""
        undistorted_k, _ = cv2.getOptimalNewCameraMatrix(
            self.k, self.d, (self.width, self.height), alpha
        )

        return CameraModel(  # TODO(amadeuszsz): handle top class properly
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


class PolynomialOpenCVCameraModel(OpenCVCameraModel):
    """Polynomial OpenCV camera model class."""

    def __init__(
        self,
        k: Optional[np.array] = None,
        d: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        super().__init__(k, d, height, width)
        self.flags |= cv2.CALIB_FIX_K1
        self.flags |= cv2.CALIB_FIX_K2
        self.flags |= cv2.CALIB_FIX_K3


class RationalOpenCVCameraModel(PolynomialOpenCVCameraModel):
    """Rational OpenCV camera model class."""

    def __init__(
        self,
        k: Optional[np.array] = None,
        d: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        super().__init__(k, d, height, width)
        self.flags |= cv2.CALIB_RATIONAL_MODEL
        self.flags |= cv2.CALIB_FIX_K4
        self.flags |= cv2.CALIB_FIX_K5
        self.flags |= cv2.CALIB_FIX_K6


class PrismOpenCVCameraModel(RationalOpenCVCameraModel):
    """Prism OpenCV camera model class."""

    def __init__(
        self,
        k: Optional[np.array] = None,
        d: Optional[np.array] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ):
        super().__init__(k, d, height, width)
        self.flags |= cv2.CALIB_THIN_PRISM_MODEL
