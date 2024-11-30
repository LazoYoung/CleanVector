from huggingface_hub import login

from backend.diffusion import DiffusionModel
from backend.util import get_token, save_image

if __name__ == "__main__":
    login(token=get_token(path='log/token'))

    model = DiffusionModel()
    samples = model.sample_images(prompt="a cat")

    # Save the image (optional)
    for img in samples:
        save_image(img, prefix="before")
