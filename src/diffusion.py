from dataclasses import dataclass
from os import makedirs

import torch
from diffusers import DDIMScheduler, AutoPipelineForText2Image
from diffusers.utils import is_xformers_available
from compel import Compel, ReturnedEmbeddingsType

from .parser import read_yaml
from .sentence_attention import AttentionScoreComputer
from .util import random_path


class DiffusionModel:
    cfg: any
    pipeline: any
    compel: Compel

    def __init__(self):
        self.read_config()
        self.compile()
        self.attention = AttentionScoreComputer()

    def read_config(self):
        self.cfg = read_yaml("resource/config/diffusion.yml")
        print("Diffusion config:", self.cfg)

    def compile(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = self.cfg.model_id
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Load pre-trained T2I model
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

        if hasattr(pipe, 'tokenizer_2'):
            self.compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True]
            )
        else:
            self.compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    def sample_images(self, prompt=None, num_sample=3, style="lineart") -> list:
        """
        Generate an image using a pre-trained diffusion model.

        Args:
            prompt: (optional) text to control image generation
            num_sample: number of samples to generate
            style: drawing style defined at diffusion.yml

        Returns:
            list: list of generated PIL.Image
        """

        if self.pipeline is None:
            self.compile()

        if prompt:
            prompt = self.parse_prompt(prompt, style)
        else:
            prompt = self.cfg.prompt

        print(f"Positive prompt: {prompt}")
        print(f"Negative prompt: {self.cfg.negative_prompt}")

        if hasattr(self.pipeline, 'tokenizer_2'):
            cond, pooled = self.compel(prompt)
            prompt_embeddings = {
                'prompt_embeds': cond,
                'pooled_prompt_embeds': pooled
            }
        else:
            prompt_embeddings = {'prompt_embeds': self.compel(prompt)}

        output = []

        for i in range(num_sample):
            with torch.inference_mode():
                sample = self.pipeline(
                    **prompt_embeddings,
                    negative_prompt=self.cfg.negative_prompt,
                    width=self.cfg.width,
                    height=self.cfg.height,
                    num_images_per_prompt=1,
                    num_inference_steps=self.cfg.num_inference_step,
                    guidance_scale=self.cfg.guidance_scale,
                )
                output.extend(sample.images)

        if self.cfg.save:
            save_dir = self.cfg.save_dir

            for image in output:
                makedirs(save_dir, exist_ok=True)
                image.save(random_path("png", save_dir))

        return output

    def parse_prompt(self, user_prompt: str, style: str) -> str:
        # subject_tokens = self.attention.get_most_relevant_tokens(user_prompt, top_k=2)
        # print("Prompt score:", subject_tokens)
        #
        # subject_tokens = dict((token, score) for token, score in subject_tokens)
        # weighted_prompt = ""
        #
        # for token in user_prompt.split():
        #     if token in subject_tokens:
        #         token += "++ "
        #     else:
        #         token += ' '
        #     weighted_prompt += token

        weighted_prompt = f"({user_prompt})1.3"

        prompt = self.cfg.styles[style].format(prompt=weighted_prompt.strip())
        return prompt
