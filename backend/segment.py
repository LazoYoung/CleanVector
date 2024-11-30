import os

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class UNet(nn.Module):
    """
    U-Net architecture for image segmentation.
    Supports flexible input and output channel configurations.
    """

    def __init__(self, in_channels=3, out_channels=1, features=64):
        super(UNet, self).__init__()

        # Encoder (Down-sampling) path
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge
        self.bottleneck = self._block(features * 8, features * 16)

        # Decoder (Up-sampling) path
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self._block((features * 8) * 2, features * 8)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self._block((features * 4) * 2, features * 4)

        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self._block((features * 2) * 2, features * 2)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self._block(features * 2, features)

        # Final convolution for output segmentation map
        self.final_conv = nn.Conv2d(
            features, out_channels, kernel_size=1
        )

    def _block(self, in_channels, features):
        """
        Double convolution block with BatchNorm and ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bridge between encoders and decoders
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))


class SegmentationDataset(Dataset):
    """
    Custom Dataset for image segmentation tasks
    Handles image and mask loading with optional transformations
    """

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
        self.mask_paths = [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)]
        self.transform = transform or self._default_transform()

    def _default_transform(self):
        return transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0., 0., 0.], std=[0., 0., 0.])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')

        # Apply transformations
        image = self.transform(image)
        mask = transforms.Compose([
            transforms.Resize((512, 512)),  # todo: mask size must be identical to resized image
            transforms.ToTensor()
        ])(mask)

        return image, mask


def train_unet(model, train_loader, criterion, optimizer, device, epochs=10):
    """
    Training loop for U-Net segmentation model
    """
    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}')


# todo extract region of interest and discard background
# find suitable dataset from https://sundong.tistory.com/7
def main():
    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 20

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create dataset and dataloader
    dataset = SegmentationDataset('path/to/images', 'path/to/masks')
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model
    model = UNet(in_channels=3, out_channels=1)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_unet(model, train_loader, criterion, optimizer, device, EPOCHS)

    # Save the model
    torch.save(model.state_dict(), 'unet_segmentation_model.pth')


def calculate_scale(dataset):
    """
    Calculate per-channel mean and standard deviation
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for image, _ in dataset:
        mean += image.mean(dim=[1, 2])
        std += image.std(dim=[1, 2])

    mean /= len(dataset)
    std /= len(dataset)

    return mean, std


def predict(model, image_path, device, mean=None, std=None):
    """
    Perform segmentation prediction on a single image
    """
    if mean is None:
        mean = [0., 0., 0.]

    if std is None:
        std = [0., 0., 0.]

    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # todo: image shall be resized to match what's been used to train U-Net
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    with torch.no_grad():
        image = transform(image).unsqueeze(0).to(device)
        segmentation = model(image)
        segmentation = segmentation.squeeze().cpu().numpy()

    return segmentation


if __name__ == '__main__':
    main()
