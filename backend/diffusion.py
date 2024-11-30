from dataclasses import dataclass

import torch
from diffusers import DDIMScheduler, AutoPipelineForText2Image
from diffusers.utils import is_xformers_available

from .parser import read_yaml


@dataclass
class DiffusionConfig:
    model_id: str
    prompt: str
    width: int
    height: int
    num_samples: int
    num_inference_step: int
    guidance_scale: float
    xformers: bool


class DiffusionModel:
    cfg: any
    pipeline: any

    def __init__(self):
        self.read_config()
        self.compile()

    def read_config(self):
        self.cfg = read_yaml("config/diffusion.yml")
        print("Diffusion config:", self.cfg)

    def compile(self):
        # Check for CUDA availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = self.cfg.model_id

        # Load scheduler
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Load the pre-trained model
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe = pipe.to(device)

        if self.cfg.xformers:
            if is_xformers_available():
                pipe.enable_xformers_memory_efficient_attention()
            else:
                print("Please install `xformers` to boost inference.")

        self.pipeline = pipe

    def sample_images(self, prompt=None) -> list:
        """
        Generate an image using a pre-trained diffusion model.

        Args:
            prompt: (optional) text to control image generation

        Returns:
            list: list of generated PIL.Image
        """

        if self.pipeline is None:
            self.compile()

        prompt = self.cfg.prompt + prompt
        output = []

        for i in range(self.cfg.num_samples):
            with torch.inference_mode():
                sample = self.pipeline(
                    prompt=prompt,
                    negative_prompt=self.cfg.negative_prompt,
                    width=self.cfg.width,
                    height=self.cfg.height,
                    num_images_per_prompt=1,
                    num_inference_steps=self.cfg.num_inference_step,
                    guidance_scale=self.cfg.guidance_scale,
                )
                output.extend(sample.images)

        return output
