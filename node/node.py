from typing import Literal
import torch
from comfy.model_management import get_torch_device
from .pipeline import SimplifiedSegmenter
import torch.nn.functional as F

class InteriorDesignNode:
    def __init__(self):
        self.device = get_torch_device()
        self.pipeline = SimplifiedSegmenter(
            device=self.device
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  
                "control_items": ("CONTROL_ITEMS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK") 
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "process"
    CATEGORY = "segmentation"

    def process(self, image, control_items):
        image = image.permute(0, 3, 1, 2).to(self.device)

        seg_cond, mask = self.pipeline(images=image,control_items=control_items) 
        seg_cond = seg_cond.permute(0, 2, 3, 1)
        mask = mask.permute(0, 2, 3, 1)
        mask = mask.squeeze(-1)

        #seg_cond = seg_cond.float() / 255.0 (its already normalized in the pipeline)

        return (seg_cond, mask,)
    

class ImageNormalize:
    def __init__(self):
        self.device = get_torch_device()
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_size": ("INT", {
                    "default": 512,
                    "min": 2,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "multiple": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 128,
                    "step": 1,
                    "display": "number",
                }),
                "mode": (["bilinear", "nearest", "bicubic", "trilinear","area","nearest-exact"],),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "resize_tensor_to_target"
    CATEGORY = "image"




    def resize_tensor_to_target(
        self,
        images: torch.Tensor,
        target_size: int,
        multiple: int,
        mode: Literal[
            "nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"
        ] = "bilinear"
    ) -> torch.Tensor:

        if images.dim() == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)

        b, c, h, w = images.shape

        max_dim = max(h, w)
        scale = target_size / max_dim if max_dim > target_size else 1.0

        new_h = int(h * scale)
        new_w = int(w * scale)

        new_h = ((new_h + multiple - 1) // multiple) * multiple
        new_w = ((new_w + multiple - 1) // multiple) * multiple

        resized = F.interpolate(
            images, size=(new_h, new_w), mode=mode,
            align_corners=False if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
        )

        resized = resized.permute(0, 2, 3, 1)

        return (resized,)


class ControlItems:
    def __init__(self):
        self.device = get_torch_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "window": ("BOOLEAN", {"default": True}),
                "door": ("BOOLEAN", {"default": True}),
                "staircase": ("BOOLEAN", {"default": False}),
                "columns": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONTROL_ITEMS",)
    RETURN_NAMES = ("CONTROL_ITEMS",)
    FUNCTION = "process"
    CATEGORY = "control"

    def process(self, window: bool, door: bool, staircase: bool, columns: bool):
        control_items = []
        if window:
            control_items.append("windowpane;window")
        if door:
            control_items.append("door;double;door")
        if staircase:
            control_items.append("stairway;staircase")
        if columns:
            control_items.append("column;pillar")
        return (control_items,) 
