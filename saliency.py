import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import ToTensor

from src.diffusion import DiffusionModel
from src.model import load_isnet
from src.util import read_images

save_dir = "models"


class Saliency:
    def __init__(self, model, mean, std, resize_shape, mapper, device):
        self.model = model
        self.mean = mean
        self.std = std
        self.resize_shape = resize_shape
        self.map = mapper  # output mapper function
        self.device = device
        self.transform = T.Compose([
            T.ToTensor(), T.Normalize(self.mean, self.std)
        ])

    def segment(self, images, img_format=True):
        """
        Compute saliency segmentation to compute attention maps.

        Args:
            images: target images
            img_format: whether to reshape the output as image-compatible (C * H * W)

        Returns: a list of salient map of shape (H * W).

        """
        self.model.eval()
        to_tensor = ToTensor()
        results = []

        for img in images:
            img = to_tensor(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to(self.device)
            with torch.no_grad():
                output = self.model(img)
            output = self.map(output).cpu()
            output = torch.squeeze(output, dim=0 if img_format else (0, 1))
            output = output.numpy()
            if img_format:
                output = self._to_image(output)
            results.append(output)
        return results

    def _to_image(self, smap):
        # invert normalization
        smap[:, :, 0] = ((smap[:, :, 0]) * self.std[0]) + self.mean[0]
        smap[:, :, 1] = ((smap[:, :, 1]) * self.std[1]) + self.mean[1]
        smap[:, :, 2] = ((smap[:, :, 2]) * self.std[2]) + self.mean[2]
        # (C * H * W) -> (H * W * C)
        image = np.transpose(smap, axes=(1, 2, 0))
        image = np.clip(image * 255., 0., 255.)
        image = image.astype(np.uint8)
        return image


class ISNet(Saliency):
    def __init__(self, device):
        def mapper(output):
            return output[0][0]

        super().__init__(
            model=load_isnet(device),
            mean=np.array([0.5, 0.5, 0.5]),
            std=np.array([1., 1., 1.]),
            resize_shape=(1024, 1024),
            mapper=mapper,
            device = device,
        )


def visualize_saliency(images):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    isnet = ISNet(device)
    saliency_maps = isnet.segment(images)

    for idx, map in enumerate(saliency_maps):
        fig, ax = plt.subplots(2, 1, figsize=(4, 8))
        fig.suptitle("Saliency map")
        ax[0].imshow(images[idx])
        ax[1].imshow(map)
        fig.tight_layout()
        plt.show()
        plt.close()


def main(args):
    if args.diffusion:
        images = DiffusionModel().sample_images(args.prompt, args.num_sample)
    else:
        images = read_images(args.file_path, args.num_sample)

    visualize_saliency(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Saliency segmentation"
    )
    parser.add_argument("-n", "--num_sample", type=int, default=5,
                        help="number of samples to use")
    parser.add_argument("-d", "--diffusion", action="store_true",
                        help="use diffusion to generate samples")
    parser.add_argument("-p", "--prompt", type=str, default="a cat face",
                        help="prompt used to control diffusion")
    parser.add_argument("-f", "--file_path", type=str, default="resource/segment/img",
                        help="path to sample directory")
    args = parser.parse_args()
    main(args)
