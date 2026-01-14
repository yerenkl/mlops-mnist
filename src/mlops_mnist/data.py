import os
from glob import glob

import torch
import typer


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    train_imgs = sorted(glob(f"{raw_dir}/train_images_*.pt"))
    train_targets = sorted(glob(f"{raw_dir}/train_target_*.pt"))

    for i in range(len(train_imgs)):
        imgs = torch.load(train_imgs[i])
        targets = torch.load(train_targets[i])
        if i == 0:
            train_data = imgs
            train_labels = targets
        else:
            train_data = torch.cat((train_data, imgs), dim=0)
            train_labels = torch.cat((train_labels, targets), dim=0)

    test_data = torch.load(f"{raw_dir}/test_images.pt")
    test_labels = torch.load(f"{raw_dir}/test_target.pt")

    # Add dimension of size 1 to specified position
    train_data = train_data.unsqueeze(1)
    test_data = test_data.unsqueeze(1)
    train_labels = train_labels.long()
    test_labels = test_labels.long()

    train_images = normalize(train_data)
    test_images = normalize(test_data)

    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    torch.save(train_images, f"{processed_dir}/train_images.pt")
    torch.save(train_labels, f"{processed_dir}/train_target.pt")
    torch.save(test_images, f"{processed_dir}/test_images.pt")
    torch.save(test_labels, f"{processed_dir}/test_target.pt")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    train_images = torch.load("data/processed/train_images.pt")
    train_target = torch.load("data/processed/train_target.pt")
    test_images = torch.load("data/processed/test_images.pt")
    test_target = torch.load("data/processed/test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__": # pragma: no cover
    typer.run(preprocess_data)
