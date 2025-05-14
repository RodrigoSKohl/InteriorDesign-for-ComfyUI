<div align="center">

# Interior Design for Comfyui

</div>

This node adapts [StableDesign: 2nd place solution for the Generative Interior Design 2024 competition](https://huggingface.co/spaces/MykolaL/StableDesign) to use directly on ComfyUI.

More about this pipeline works, see this post on [Medium](https://medium.com/@melgor89/generative-interior-design-challenge-2024-2nd-place-solution-6338f19f6fe3)
#
The pipeline automatically downloads trained models of `controlnet_depth` and `own_controlnet` to the local script of ComfyUI runs in folder `./custom_models`. Other models are downloaded to the `.cache` folder of Hugging Face.

This node is available on [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
![image](https://github.com/user-attachments/assets/44e3520c-1b42-41a2-8996-47e5fb37d9a1)
