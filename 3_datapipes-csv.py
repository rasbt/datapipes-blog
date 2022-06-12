import os

import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader, default_collate
from torchdata import datapipes as dp
from torchvision import transforms
from watermark import watermark

IMG_ROOT = "mnist-pngs"

DATA_TRANSFORMS = {
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


def create_path_label_pair(inputs):
    img_path, label = inputs
    img_path = os.path.join(IMG_ROOT, img_path)
    label = int(label)
    return img_path, label


def open_image(inputs):
    img_path, label = inputs
    img = Image.open(img_path)
    return img, label


def apply_train_transforms(inputs):
    x, y = inputs
    return DATA_TRANSFORMS["train"](x), y


def apply_test_transforms(inputs):
    x, y = inputs
    return DATA_TRANSFORMS["test"](x), y


def build_data_pipe(csv_file, transform, len=1000, batch_size=32):
    new_dp = dp.iter.FileOpener([csv_file])

    new_dp = new_dp.parse_csv(skip_lines=1)
    # returns tuples like ('train/0/16585.png', '0')

    new_dp = new_dp.map(create_path_label_pair)
    # returns tuples like ('mnist-pngs/train/0/16585.png', 0)

    if transform == "train":
        new_dp = new_dp.shuffle(buffer_size=len)

    new_dp = new_dp.map(open_image)

    if transform == "train":
        new_dp = new_dp.map(apply_train_transforms)
        new_dp = new_dp.batch(batch_size=batch_size, drop_last=True)

    elif transform == "test":
        new_dp = new_dp.map(apply_test_transforms)
        new_dp = new_dp.batch(batch_size=batch_size, drop_last=False)

    else:
        raise ValueError("Invalid transform argument.")

    new_dp = new_dp.map(default_collate)
    return new_dp


if __name__ == "__main__":

    print(watermark(packages="torch,torchdata", python=True))

    train_dp = build_data_pipe(
        csv_file="mnist-pngs/new_train.csv", transform="train", len=45000, batch_size=32
    )

    val_dp = build_data_pipe(
        csv_file="mnist-pngs/new_val.csv", transform="test", batch_size=32
    )

    test_dp = build_data_pipe(
        csv_file="mnist-pngs/test.csv", transform="test", batch_size=32
    )

    train_loader = DataLoader(dataset=train_dp, shuffle=True, num_workers=2)

    val_loader = DataLoader(dataset=val_dp, shuffle=False, num_workers=2)

    test_loader = DataLoader(dataset=test_dp, shuffle=False, num_workers=2)

    num_epochs = 1
    for epoch in range(num_epochs):

        for batch_idx, (x, y) in enumerate(train_loader):
            if batch_idx >= 3:
                break

            # collate added an extra dimension
            x, y = x[0], y[0]
            print(" Batch index:", batch_idx, end="")
            print(" | Batch size:", y.shape[0], end="")
            print(" | x shape:", x.shape, end="")
            print(" | y shape:", y.shape)

    print("Labels from current batch:", y)

    # Uncomment to visualize a data batch:
    # batch = next(iter(train_loader))
    # viz_batch_images(batch[0])
