from typing import Tuple,List, Union
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation, AutoModelForDepthEstimation
from .colors import ade_palette
from .utils import map_colors_rgb
import torch
from torch import Tensor
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

# -------------------
# Classe principal
# -------------------

class SimplifiedSegmenter:
    def __init__(self, device: str = "cuda"):
        self.device = device

        print("🔄 Loading segmentator...")
        self.image_processor = AutoImageProcessor.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-small"
        )
        self.image_segmentor.eval()

    def tensor_to_pil(self, tensor: Tensor) -> Image.Image:
        if tensor.max() > 1.0:
            arr = tensor.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            arr = (tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr)

    def segment_image(self,
            image: Image,
    ) -> Image:
        """
        Segments an image using a semantic segmentation model.
        Args:
            image (Image): The input image to be segmented.
            image_processor (AutoImageProcessor): The processor to prepare the
                image for segmentation.
            image_segmentor (UperNetForSemanticSegmentation): The semantic
                segmentation model used to identify different segments in the image.
        Returns:
            Image: The segmented image with each segment colored differently based
                on its identified class.
        """
        # image_processor, image_segmentor = get_segmentation_pipeline()
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        image_size = image.size[::-1]
        print(f"Image size: {image_size}, pixel_values shape: {pixel_values.shape}")
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)
       
        seg = self.image_processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image_size])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        palette = np.array(ade_palette())
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        seg_image = Image.fromarray(color_seg).convert('RGB')
        return seg_image
    
    @staticmethod
    def filter_items(
        colors_list: Union[List, np.ndarray],
        items_list: Union[List, np.ndarray],
        items_to_remove: Union[List, np.ndarray]
    ) -> Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]:
        """
        Filters items and their corresponding colors from given lists, excluding
        specified items.
        Args:
            colors_list: A list or numpy array of colors corresponding to items.
            items_list: A list or numpy array of items.
            items_to_remove: A list or numpy array of items to be removed.
        Returns:
            A tuple of two lists or numpy arrays: filtered colors and filtered
            items.
        """
        filtered_colors = []
        filtered_items = []
        for color, item in zip(colors_list, items_list):
            if item not in items_to_remove:
                filtered_colors.append(color)
                filtered_items.append(item)
        return filtered_colors, filtered_items
    

    def __call__(self, images: Tensor, control_items: List) -> Tuple[Tensor, Tensor]:
        """
        images: Tensor (B,C,H,W) ou (C,H,W)
        Returns:
          seg_cond_tensor: (B,3,H,W) uint8
          mask_tensor: (B,1,H,W) float 0/1
        """
        print(f'Control items: {control_items}')
        if images.dim() == 4:
            input_image = self.tensor_to_pil(images[0])
        else:
            input_image = self.tensor_to_pil(images)
        real_seg = np.array(self.segment_image(input_image))
        unique_colors = np.unique(real_seg.reshape(-1, real_seg.shape[2]), axis=0)
        unique_colors = [tuple(color) for color in unique_colors]
        segment_items = [map_colors_rgb(i) for i in unique_colors]
        chosen_colors, segment_items = self.filter_items(
            colors_list=unique_colors,
            items_list=segment_items,
            items_to_remove=control_items
        )
        mask = np.zeros((real_seg.shape[0], real_seg.shape[1]), dtype=np.uint8)
        for color in chosen_colors:
            color_matches = (real_seg == color).all(axis=2)
            mask[color_matches] = 1

        seg_cond_image = Image.fromarray(real_seg).convert("RGB")
        seg_cond_tensor = torch.from_numpy(np.array(seg_cond_image)).permute(2,0,1).unsqueeze(0).to(self.device).float() / 255.0 # shape (1,3,H,W)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()  # shape (1,1,H,W)
        mask_tensor = mask_tensor.to('cpu')  # move to cpu (avoid bug when use the mask on IP Adapter?)
        print(f'Shape of seg_cond_tensor: {seg_cond_tensor.shape}, mask_tensor: {mask_tensor.shape}')

        return seg_cond_tensor, mask_tensor
