import argparse
from os import makedirs

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.unet import SegmentationDataset, UNet, train_unet
from src.util import random_path

save_dir = "models"


# todo extract region of interest and discard background
# find suitable dataset from https://sundong.tistory.com/7
def train(args):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader
    dataset = SegmentationDataset(image_dir=args.image_dir, mask_dir=args.mask_dir)
    train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    # Initialize model
    model = UNet(in_channels=3, out_channels=1)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Train the model
    train_unet(model, train_loader, criterion, optimizer, device, args.epoch)

    # Save the model
    save(model, args)


def save(model, args):
    if args.save_file:
        path = f"{save_dir}/{args.save_file}"
    else:
        path = random_path(ext="pth", dir=save_dir, prefix="segment")
    makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    parser = argparse.ArgumentParser(
        description="Segmentation training"
    )
    parser.add_argument("-b", "--batch", type=int, default=8,
                        help="batch size")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-4,
                        help="learning rate")
    parser.add_argument("-e", "--epoch", type=int, default=20,
                        help="number of epochs to train")
    parser.add_argument("-f", "--save_file", type=str, default=None,
                        help="file used to save this model")
    parser.add_argument("-i", "--image_dir", type=str, default="dataset/segment/img",
                        help="path to image dataset")
    parser.add_argument("-m", "--mask_dir", type=str, default="dataset/segment/mask",
                        help="path to mask dataset")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
