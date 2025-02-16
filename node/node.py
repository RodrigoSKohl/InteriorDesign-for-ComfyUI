import torch
import torch.nn.functional as F
from comfy.model_management import get_torch_device
from .pipeline import ControlNetDepthDesignModelMulti  # Seu pipeline original
from torchvision import transforms
from torchvision.transforms import ToPILImage

class InteriorDesignNode:
    def __init__(self):
        self.device = get_torch_device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.pipeline = ControlNetDepthDesignModelMulti()  # Seu pipeline original

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  
                "positive": ("STRING",{"forceInput": True}),  # Prompt de descrição do design
                "guidance_scale": ("FLOAT", {"default": 10, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "strength": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0}),
                "seed": ("INT", {"default": 40, "min": 0, "max": 0xffffffffffffffff}),
                "img_size": ("INT", {"default": 768, "min": 256, "max": 768, "step": 8}),
                "img_dept_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.99, "step": 0.01}),
                "img_seg_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.99, "step": 0.01}),
            },
            "optional": {
                "negative": ("STRING",{"forceInput": True}),  # Prompt de descrição do design
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Retorna a imagem gerada
    RETURN_NAMES = ("IMAGE",)  # Nome do retorno

    FUNCTION = "process"  # Função que será executada
    CATEGORY = "interior_design"  # Categoria para o ComfyUI

    def process(self, image, positive, guidance_scale, num_steps, strength, seed, img_size, img_dept_threshold, img_seg_threshold, negative=None):
        # Pré-processamento da imagem
        self.pipeline.seed = seed
        if negative is not None:
            self.pipeline.neg_prompt = negative
        image = tensor_to_pil(image)
        
        # Chama o pipeline para gerar o design de interiores
        design_image = self.pipeline.generate_design(
            empty_room_image=image,
            prompt=positive,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            strength=strength,
            img_size=img_size,
            img_dept_threshold=img_dept_threshold,
            img_seg_threshold=img_seg_threshold,
        )
        design_image = pil_to_tensor(design_image, self.device, self.dtype)
        return (design_image,)

def tensor_to_pil(image_tensor):
    print(f"Tensor: {image_tensor.shape},{image_tensor.dtype}")
    if image_tensor.ndimension() == 4:
        image_tensor = image_tensor.squeeze(0)  # Remove a dimensão do batch
    image_tensor = image_tensor.permute(2, 0, 1)  # Troca de [altura, largura, canais] -> [canais, altura, largura]
    if image_tensor.shape[0] == 4: 
        image_tensor = image_tensor[:3]
    to_pil = ToPILImage()
    pil_image = to_pil(image_tensor)
    print(f"Imagem PIL: {pil_image.size},{pil_image.mode}")
    return pil_image

def pil_to_tensor(image_pil, device, dtype):
    print(f"Imagem PIL: {image_pil.size},{image_pil.mode}")
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(image_pil)
    tensor = tensor.permute(1, 2, 0)  # Troca de [canais, altura, largura] -> [altura, largura, canais]
    tensor = tensor.unsqueeze(0)  # Adiciona a dimensão do batch
    tensor = tensor.to(device, dtype=dtype)
    print(f"Tensor: {tensor.shape},{tensor.dtype}")
    return tensor