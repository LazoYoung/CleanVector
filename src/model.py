import os

import requests
import torch

from src.isnet import ISNetDIS

model_path = "resource/model"



def load_model_path(url, filename):
    """
    Download the model from URL if it doesn't exist locally.

    Args:
    url (str): The URL of the file to download
    local_filename (str, optional): The local path to save the file.

    Returns:
    str: The local path of the downloaded or existing file
    """

    filename = os.path.join(model_path, filename)

    if os.path.exists(filename):
        return filename

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        print(f"Downloading {url}")
        with requests.get(url, stream=True) as response:
            # Raise an exception for HTTP errors
            response.raise_for_status()

            # Open the local file to write the downloaded content
            with open(filename, 'wb') as file:
                # Write the file in chunks to handle large files efficiently
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        print("Done.")
        return filename

    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        return None




def load_isnet(device):
    path = load_model_path(
        url="https://huggingface.co/NimaBoscarino/IS-Net_DIS-general-use/resolve/main/isnet-general-use.pth",
        filename="isnet-general-use.pth"
    )
    model = ISNetDIS()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model = model.to(device)

    return model
