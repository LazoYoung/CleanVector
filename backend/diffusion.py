from dataclasses import dataclass

import torch
from PIL import Image
from diffusers import DDIMScheduler, AutoPipelineForText2Image
from diffusers.utils import is_xformers_available

from backend.yaml import YAMLDeserializer


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


def get_config():
    return YAMLDeserializer().from_yaml_file("config/diffusion.yml", DiffusionConfig)


class DiffusionModel:
    def __init__(self):
        self.cfg = get_config()
        self.pipeline = None
        self.compile()

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

    def sample_images(self) -> list:
        """
        Generate an image using a pre-trained diffusion model.

        Returns:
            list: List of generated PIL.Image
        """

        if self.pipeline is None:
            self.compile()

        output = []

        for i in range(self.cfg.num_samples):
            with torch.inference_mode():
                sample = self.pipeline(
                    prompt=self.cfg.prompt,
                    width=self.cfg.width,
                    height=self.cfg.height,
                    num_images_per_prompt=1,
                    num_inference_steps=self.cfg.num_inference_step,
                    guidance_scale=self.cfg.guidance_scale,
                )
                output.extend(sample.images)

        return output
