import argparse

from src.diffusion import DiffusionModel
from src.image_refiner import refine
from src.util import authenticate, measure, Time, read_images


def diffusion(args):
    authenticate(args.token)

    model = DiffusionModel()

    if args.style:
        style = args.style
    else:
        print("Available style:", list(model.cfg.styles.keys()))
        style = input("Select one... ")

    def sample_images():
        return model.sample_images(prompt=args.prompt, num_sample=args.num_sample, style=style)

    elapsed_time, samples = measure(sample_images, Time.SECOND)
    print(f"Generated {len(samples)} samples in {elapsed_time:.1f} sec.")
    return samples


def main(args):
    if args.num_sample > 0:
        diffusion(args)
    else:
        print("Skipping diffusion step...")

    refine(read_images("output/diffused"), args.prompt, "output/cropped", detect_objects=args.detr, prompt_threshold=args.prompt_threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate bitmap images with diffusion."
    )
    parser.add_argument("-p", "--prompt", type=str, required=True,
                        help="prompt used to control sampling.")
    parser.add_argument("-n", "--num_sample", type=int, default=3,
                        help="number of samples to generate")
    parser.add_argument("-s", "--style", type=str, default="icon",
                        help="Style of image: cinematic, anime, photographic, comic, icon, lineart, pixelart")
    parser.add_argument("-t", "--token", type=str, default=None,
                        help="set huggingface access token.")
    parser.add_argument("--detr", action="store_true",
                        help="use DETR for object detection and image cropping")
    parser.add_argument("--prompt_threshold", type=float, default=0.3,
                        help="DETR threshold of minimum similarity between an object and the prompt")
    parser.add_argument("--token_path", type=str, default=".token",
                        help="set path to token storage file.")
    args = parser.parse_args()
    main(args)
