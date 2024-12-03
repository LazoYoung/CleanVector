import argparse

from src.diffusion import DiffusionModel
from src.util import authenticate, measure, Time


def infer(args):
    def sample_images():
        return model.sample_images(prompt=args.prompt, num_sample=args.num_sample)

    authenticate(args.token)

    print(f"Prompt: {args.prompt}")

    model = DiffusionModel()
    elapsed_time, samples = measure(sample_images, Time.SECOND)

    print(f"Generated {len(samples)} samples in {elapsed_time:.1f} sec.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate bitmap images with diffusion."
    )
    parser.add_argument("-p", "--prompt", type=str, default=None,
                        help="prompt used to control sampling.")
    parser.add_argument("-n", "-num_sample", type=int, default=3,
                        help="number of samples to generate")
    parser.add_argument("-t", "--token", type=str, default=None,
                        help="set huggingface access token.")
    parser.add_argument("--token_path", type=str, default=".token",
                        help="set path to token storage file.")
    args = parser.parse_args()
    infer(args)


if __name__ == "__main__":
    main()
