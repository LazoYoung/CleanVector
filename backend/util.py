import uuid
from datetime import datetime
from os import mkdir, makedirs

from PIL import Image


def get_token(path):
    with open(path, 'r') as file:
        token = file.read().strip()
    return token


def save_image(image: Image, dir="images", prefix=None):
    unique_id = str(uuid.uuid4())[:8]
    makedirs(dir, exist_ok=True)

    if prefix:
        filename = f"{dir}/{prefix}_{unique_id}.png"
    else:
        filename = f"{dir}/{unique_id}.png"

    image.save(filename)