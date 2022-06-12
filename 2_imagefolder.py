import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from watermark import watermark


def viz_batch_images(batch):

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training images")
    plt.imshow(
        np.transpose(
            vutils.make_grid(batch[0][:64], padding=2, normalize=True), (1, 2, 0)
        )
    )
    plt.show()


if __name__ == "__main__":

    print(watermark(packages="torch,torchdata", python=True))

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize(32),
                transforms.RandomCrop((28, 28)),
                transforms.ToTensor(),
                # normalize images to [-1, 1] range
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(32),
                transforms.CenterCrop((28, 28)),
                transforms.ToTensor(),
                # normalize images to [-1, 1] range
                transforms.Normalize((0.5,), (0.5,)),
            ]
        ),
    }

    train_dataset = ImageFolder(
        root="mnist-pngs/train", transform=data_transforms["train"]
    )

    train_dataset, val_dataset = random_split(train_dataset, lengths=[55000, 5000])

    test_dataset = ImageFolder(
        root="mnist-pngs/test", transform=data_transforms["test"]
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,  # want to shuffle the dataset
        num_workers=2,
    )  # number processes/CPUs to use

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=True,  # want to shuffle the dataset
        num_workers=2,
    )  # number processes/CPUs to use

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=True,  # want to shuffle the dataset
        num_workers=2,
    )  # number processes/CPUs to use

    num_epochs = 1
    for epoch in range(num_epochs):

        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= 3:
                break
            print(" Batch index:", batch_idx, end="")
            print(" | Batch size:", y.shape[0], end="")
            print(" | x shape:", x.shape, end="")
            print(" | y shape:", y.shape)

    print("Labels from current batch:", y)

    # Uncomment to visualize a data batch:
    # batch = next(iter(train_loader))
    # viz_batch_images(batch[0])
