import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset import get_food_dataset, RemappedSubset
from torchvision import transforms

CLASS_NAMES = [
    'apple_pie', 'caesar_salad', 'clam_chowder', 'edamame',
    'french_fries', 'hamburger', 'hot_dog', 'ice_cream', 'sushi', 'waffles'
]

# ImageNet normalisation — same as in get_loaders
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict_samples(model, device, output_path, name='predictions', seed=42):
    rng = np.random.default_rng(seed)

    raw_train, raw_test, _, test_indices, label_mapping = get_food_dataset()
    test_dataset = RemappedSubset(raw_test, test_indices, label_mapping, transform=_TRANSFORM)

    # Pick one random sample per class (10 classes)
    labels_array = np.array([label_mapping[raw_test._labels[i]] for i in test_indices])
    selected_indices = []
    for cls in range(len(CLASS_NAMES)):
        cls_positions = np.where(labels_array == cls)[0]
        selected_indices.append(int(rng.choice(cls_positions)))

    model.eval()

    fig, axes = plt.subplots(2, 10, figsize=(22, 6))

    with torch.no_grad():
        for col, idx in enumerate(selected_indices):
            image, true_label = test_dataset[idx]

            input_tensor = image.unsqueeze(0).to(device)
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_label = int(probs.argmax())

            # Denormalise image for display
            display_img = (image.cpu() * _STD + _MEAN).permute(1, 2, 0).clamp(0, 1).numpy()

            axes[0][col].imshow(display_img)
            axes[0][col].set_title(
                f'T: {CLASS_NAMES[true_label]}\nP: {CLASS_NAMES[pred_label]}',
                color='green' if pred_label == true_label else 'red',
                fontsize=7
            )
            axes[0][col].axis('off')

            axes[1][col].bar(range(len(CLASS_NAMES)), probs, color='steelblue')
            axes[1][col].set_xticks(range(len(CLASS_NAMES)))
            axes[1][col].set_xticklabels(
                [c[:4] for c in CLASS_NAMES], rotation=90, fontsize=6
            )
            axes[1][col].set_ylim(0, 1)
            axes[1][col].tick_params(labelsize=6)

    axes[1][0].set_ylabel('Probability')

    fig.suptitle('Sample Predictions — one per class (green = correct, red = wrong)', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f'{name}.png'))
    plt.close(fig)
