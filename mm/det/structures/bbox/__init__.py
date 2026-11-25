# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_overlaps import bbox_overlaps
from .base_boxes import BaseBoxes
from .horizontal_boxes import HorizontalBoxes
from .box_type import autocast_box_type, get_box_type
from .transforms import (
    bbox2roi,
    bbox_flip,
    bbox_mapping_back,
    cat_boxes,
    empty_box_as,
    get_box_tensor,
    get_box_wh,
    scale_boxes,
    stack_boxes,
)
