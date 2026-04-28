import numpy as np

from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

class RemappedSubset(Dataset):
    def __init__(self, dataset, sample_indices, label_mapping, transform=None):
        self.dataset = Subset(dataset, sample_indices)
        self.sample_indices = sample_indices
        self.label_mapping = label_mapping
        self.transform = transform

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        image, old_label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        new_label = self.label_mapping[old_label]
        return image, new_label

def get_food_dataset():
    target_classes = [
        'apple_pie',
        'caesar_salad',
        'clam_chowder',
        'edamame',
        'french_fries',
        'hamburger',
        'hot_dog',
        'ice_cream',
        'sushi',
        'waffles'
    ]

    train_dataset = datasets.Food101(
        root='./data',
        split='train',
        download=True
    )

    test_dataset = datasets.Food101(
        root='./data',
        split='test',
        download=True
    )

    target_class_labels = {train_dataset.class_to_idx[cls] for cls in target_classes}

    train_sample_indices = [i for i, label in enumerate(train_dataset._labels) if label in target_class_labels]
    test_sample_indices  = [i for i, label in enumerate(test_dataset._labels)  if label in target_class_labels]

    label_mapping = {old: new for new, old in enumerate(sorted(target_class_labels))}

    return train_dataset, test_dataset, train_sample_indices, test_sample_indices, label_mapping

def get_loaders(batch_size=32, val_ratio=0.2, seed=42, augment=True):

    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    raw_train, raw_test, train_indices, test_indices, label_mapping = get_food_dataset()

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_indices).tolist()
    split = int(len(shuffled) * (1 - val_ratio))
    train_idx, val_idx = shuffled[:split], shuffled[split:]

    active_train_transform = train_transform if augment else basic_transform

    train_dataset = RemappedSubset(raw_train, train_idx, label_mapping, transform=active_train_transform)
    val_dataset   = RemappedSubset(raw_train, val_idx,   label_mapping, transform=basic_transform)
    test_dataset  = RemappedSubset(raw_test,  test_indices, label_mapping, transform=basic_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, test_loader