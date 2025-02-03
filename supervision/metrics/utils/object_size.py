from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.core import Detections
from supervision.metrics.core import MetricTarget

if TYPE_CHECKING:
    from supervision.detection.core import Detections

SIZE_THRESHOLDS = (32**2, 96**2)


class ObjectSizeCategory(Enum):
    ANY = -1
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


def get_object_size_category(
    data: npt.NDArray, metric_target: MetricTarget
) -> npt.NDArray[np.int_]:
    """
    Get the size category of an object. Distinguish based on the metric target.

    Args:
        data (np.ndarray): The object data, shaped (N, ...).
        metric_target (MetricTarget): Determines whether boxes, masks or
            oriented bounding boxes are used.

    Returns:
        (np.ndarray) The size category of each object, matching
        the enum values of ObjectSizeCategory. Shaped (N,).
    """
    if metric_target == MetricTarget.BOXES:
        return get_bbox_size_category(data)
    if metric_target == MetricTarget.MASKS:
        return get_mask_size_category(data)
    if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
        return get_obb_size_category(data)
    raise ValueError("Invalid metric type")


def get_bbox_size_category(xyxy: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
    """
    Get the size category of a bounding boxes array.

    Args:
        xyxy (np.ndarray): The bounding boxes array shaped (N, 4).

    Returns:
        (np.ndarray) The size category of each bounding box.
    """
    if xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ValueError("Bounding boxes must be shaped (N, 4)")

    width = xyxy[:, 2] - xyxy[:, 0]
    height = xyxy[:, 3] - xyxy[:, 1]
    areas = width * height
    return _get_size_category_from_areas(areas)


def get_mask_size_category(mask: npt.NDArray[np.bool_]) -> npt.NDArray[np.int_]:
    """
    Get the size category of detection masks.

    Args:
        mask (np.ndarray): The mask array shaped (N, H, W).

    Returns:
        (np.ndarray) The size category of each mask.
    """
    if mask.ndim != 3:
        raise ValueError("Masks must be shaped (N, H, W)")

    areas = np.sum(mask, axis=(1, 2)).astype(np.float32)
    return _get_size_category_from_areas(areas)


def get_obb_size_category(xyxyxyxy: npt.NDArray[np.float32]) -> npt.NDArray[np.int_]:
    """
    Get the size category of an oriented bounding boxes array.

    Args:
        xyxyxyxy (np.ndarray): The oriented bounding boxes array shaped (N, 4, 2).

    Returns:
        (np.ndarray) The size category of each oriented bounding box.
    """
    if xyxyxyxy.ndim != 3 or xyxyxyxy.shape[1] != 4 or xyxyxyxy.shape[2] != 2:
        raise ValueError("Oriented bounding boxes must be shaped (N, 4, 2)")

    x = xyxyxyxy[:, :, 0]
    y = xyxyxyxy[:, :, 1]
    # Shoelace formula using np.roll for a more concise vectorized computation.
    areas = 0.5 * np.abs(
        np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    )
    return _get_size_category_from_areas(areas)


def get_detection_size_category(
    detections: Detections, metric_target: MetricTarget = MetricTarget.BOXES
) -> npt.NDArray[np.int_]:
    """
    Get the size category of a detections object.

    Args:
        detections (Detections): The detection object containing boxes, masks, etc.
        metric_target (MetricTarget): Determines whether boxes, masks, or oriented boxes are used.

    Returns:
        (np.ndarray) The size category of each detection.
    """
    if metric_target == MetricTarget.BOXES:
        return get_bbox_size_category(detections.xyxy)
    if metric_target == MetricTarget.MASKS:
        if detections.mask is None:
            raise ValueError("Detections mask is not available")
        return get_mask_size_category(detections.mask)
    if metric_target == MetricTarget.ORIENTED_BOUNDING_BOXES:
        if detections.data.get(ORIENTED_BOX_COORDINATES) is None:
            raise ValueError("Detections oriented bounding boxes are not available")
        return get_obb_size_category(
            np.array(detections.data[ORIENTED_BOX_COORDINATES])
        )
    raise ValueError("Invalid metric type")


# We assume ObjectSizeCategory and SIZE_THRESHOLDS are defined in the internal codebase.
# For example:
# from supervision.metrics.utils.object_size import SIZE_THRESHOLDS
# and ObjectSizeCategory is an enum exposing .SMALL, .MEDIUM, .LARGE, .ANY with a .value attribute.


def _get_size_category_from_areas(
    areas: npt.NDArray[np.float32],
) -> npt.NDArray[np.int_]:
    SM, LG = SIZE_THRESHOLDS
    # All areas that are less than SM become SMALL; if less than LG then MEDIUM; otherwise LARGE.
    return np.where(
        areas < SM,
        ObjectSizeCategory.SMALL.value,
        np.where(
            areas < LG, ObjectSizeCategory.MEDIUM.value, ObjectSizeCategory.LARGE.value
        ),
    )
