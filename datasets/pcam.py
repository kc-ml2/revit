import pathlib
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class PCam(ImageFolder):
    """
    PCam dataset.

    Download the dataset from https://drive.google.com/file/d/1PcPdBOyImivBz3IMYopIizGvJOnfgXGD/view?usp=sharing

    For more information, please refer to the README.md of the repository.
    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False, valid=False
    ):
        if train and valid:
            raise ValueError("PCam 'valid' split available only when train=False.")

        root = pathlib.Path(root) / "PCam"
        split = "train" if train else ("valid" if valid else "test")
        directory = root / split
        if not (root.exists() and directory.exists()):
            raise FileNotFoundError(
                "Please download the PCam dataset. How to download it can be found in 'README.md'"
            )

        super().__init__(root=directory, transform=transform, target_transform=target_transform)


def get_pcam_loaders(root="./data", batch_size=64, num_workers=8, pin_memory=True):
    
    data_mean = (0.701, 0.538, 0.692)
    data_stddev = (0.235, 0.277, 0.213)
    transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_stddev),
            ]
    )
    
    train_dataset = PCam(root, train=True, download=False, transform=transform)
    test_dataset = PCam(root, train=False, valid=True, download=False, transform=transform)

   
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
