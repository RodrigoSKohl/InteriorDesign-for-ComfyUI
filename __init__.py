from .node.node import InteriorDesignNode
from .node.node import ImageNormalize
from .node.node import ControlItems


NODE_CLASS_MAPPINGS = {
    "Interior Design Segmentator": InteriorDesignNode,
    "Image Normalize": ImageNormalize,
    "Control Items": ControlItems,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Interior Design Segmentator": "Interior Design Segmentator",
    "Image Normalize": "Image Resize Normalizer",
    "Control Items": "Control Items Selector",
}


print("\033[34mComfyUI Custom Nodes: \033[92mLoaded Interior Design for ComfyUI\033[0m")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']