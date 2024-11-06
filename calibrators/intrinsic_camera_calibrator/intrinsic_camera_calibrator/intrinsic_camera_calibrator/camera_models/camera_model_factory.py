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


from intrinsic_camera_calibrator.camera_models.opencv_camera_model import OpenCVCameraModelEnum
from intrinsic_camera_calibrator.camera_models.opencv_camera_model import OpenCVCameraModel
from intrinsic_camera_calibrator.camera_models.opencv_camera_model import PolynomialOpenCVCameraModel
from intrinsic_camera_calibrator.camera_models.opencv_camera_model import RationalOpenCVCameraModel
from intrinsic_camera_calibrator.camera_models.opencv_camera_model import PrismOpenCVCameraModel
from intrinsic_camera_calibrator.camera_models.ceres_camera_model import CeresCameraModelEnum
from intrinsic_camera_calibrator.camera_models.ceres_camera_model import CeresCameraModel
from intrinsic_camera_calibrator.camera_models.ceres_camera_model import PolynomialCeresCameraModel
from intrinsic_camera_calibrator.camera_models.ceres_camera_model import RationalCeresCameraModel


def make_opencv_camera_model(camera_model_type: OpenCVCameraModelEnum, **kwargs) -> OpenCVCameraModel:
    """Create a OpenCV camera model using a factory design pattern."""
    classes_dic = {
        OpenCVCameraModelEnum.OPENCV_POLYNOMIAL: PolynomialOpenCVCameraModel,
        OpenCVCameraModelEnum.OPENCV_RATIONAL: RationalOpenCVCameraModel,
        OpenCVCameraModelEnum.OPENCV_PRISM: PrismOpenCVCameraModel
    }
    return classes_dic[camera_model_type](**kwargs)


def make_ceres_camera_model(camera_model_type: CeresCameraModelEnum, **kwargs) -> CeresCameraModel:
    """Create a Ceres camera model using a factory design pattern."""
    classes_dic = {
        CeresCameraModelEnum.CERES_POLYNOMIAL: PolynomialCeresCameraModel,
        CeresCameraModelEnum.CERES_RATIONAL: RationalCeresCameraModel
    }
    return classes_dic[camera_model_type](**kwargs)
