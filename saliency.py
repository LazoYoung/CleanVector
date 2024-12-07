import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T

from src.diffusion import DiffusionModel
from src.model import load_isnet
from src.util import read_images

save_dir = "models"


class Saliency:
    def __init__(self, model, mean, std, mapper, device):
        self.model = model
        self.mean = mean
        self.std = std
        self.map = mapper  # output mapper function
        self.device = device
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std),
        ])

    def segment(self, images):
        """
        Compute saliency segmentation to compute attention maps.

        Args:
            images: target images

        Returns: a list of salient map of shape (C, H * W).

        """
        self.model.eval()
        results = []

        for img in images:
            img = self.transforms(img)
            img = torch.unsqueeze(img, dim=0)
            img = img.to(self.device)
            with torch.no_grad():
                output = self.model(img)
            output = self.map(output).cpu()
            output = torch.squeeze(output, dim=0)
            output = output.numpy()
            output = self._inverse_normalize(output)
            output = self._reshape(output)
            results.append(output)
        return results

    def _inverse_normalize(self, smap):
        smap[:, :, 0] = ((smap[:, :, 0]) * self.std[0]) + self.mean[0]
        smap[:, :, 1] = ((smap[:, :, 1]) * self.std[1]) + self.mean[1]
        smap[:, :, 2] = ((smap[:, :, 2]) * self.std[2]) + self.mean[2]
        return smap

    @staticmethod
    def _reshape(smap):
        # (C * H * W) -> (H * W * C)
        image = np.transpose(smap, axes=(1, 2, 0))
        image = np.clip(image, 0., 1.)
        return image


class ISNet(Saliency):
    def __init__(self, device):
        def mapper(output):
            return output[0][0]

        super().__init__(
            model=load_isnet(device),
            mean=np.array([0.5, 0.5, 0.5]),
            std=np.array([1., 1., 1.]),
            # resize_shape=(1024, 1024),
            mapper=mapper,
            device = device,
        )


def visualize(raw_images, processed_images):
    for raw_img, proc_img in zip(raw_images, processed_images):
        fig, ax = plt.subplots(2, 1, figsize=(4, 8))
        fig.suptitle("Saliency map")
        ax[0].imshow(raw_img)
        ax[1].imshow(proc_img)
        fig.tight_layout()
        plt.show()
        plt.close()


def filter_images(images, filters):
    output = []

    for image, mask in zip(images, filters):
        image = np.array(image)
        background = 255 * np.ones_like(image)
        filtered = mask * image + (1 - mask) * background
        filtered = filtered.astype(np.uint8)
        output.append(filtered)

    return output


def main(args):
    if args.diffusion:
        images = DiffusionModel().sample_images(args.prompt, args.num_sample)
    else:
        images = read_images(args.file_path, args.num_sample)

    if len(images) == 0:
        print(f"No images found at {args.file_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    isnet = ISNet(device)
    saliency_maps = isnet.segment(images)

    if args.map:
        visualize(images, saliency_maps)
    else:
        visualize(images, filter_images(images, saliency_maps))


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
    parser.add_argument("-m", "--map", action="store_true",
                        help="show saliency heatmap")
    args = parser.parse_args()
    main(args)
