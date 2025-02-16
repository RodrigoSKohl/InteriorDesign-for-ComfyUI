from .node.node import InteriorDesignNode


# Registra o nó no ComfyUI
NODE_CLASS_MAPPINGS = {
    "interior-design-for-comfyui": InteriorDesignNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "interior-design-for-comfyui": "Interior Design for ComfyUI",
}


print("\033[34mComfyUI Custom Nodes: \033[92mLoaded Interior Design for ComfyUI\033[0m")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']