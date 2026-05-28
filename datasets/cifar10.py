from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

def get_cifar10_loaders(root="./data/cifar10", batch_size=128, num_workers=8, pin_memory=True):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = datasets.CIFAR10(
    root=root,
    train=True,
    download=True,
    transform=train_transform
    )

    cifar10_train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
    )
        
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=test_transform
    )

    cifar10_test_loader = DataLoader(
        test_set,
        batch_size=batch_size*2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return cifar10_train_loader, cifar10_test_loader