import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay

def plot_metrics(dir_path, train_loss, train_acc, val_loss, val_acc, name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0][0].plot(train_loss, label='Train Loss')
    axes[0][0].set_xlabel('Epochs')
    axes[0][0].set_title('Train Loss')
    axes[0][0].grid(True)
    axes[0][0].legend()

    axes[0][1].plot(train_acc, label='Train Accuracy', color='orange')
    axes[0][1].set_xlabel('Epochs')
    axes[0][1].set_title('Train Accuracy')
    axes[0][1].grid(True)
    axes[0][1].legend()

    axes[1][0].plot(val_loss, label='Validation Loss')
    axes[1][0].set_xlabel('Epochs')
    axes[1][0].set_title('Validation Loss')
    axes[1][0].grid(True)
    axes[1][0].legend()

    axes[1][1].plot(val_acc, label='Validation Accuracy', color='orange')
    axes[1][1].set_xlabel('Epochs')
    axes[1][1].set_title('Validation Accuracy')
    axes[1][1].grid(True)
    axes[1][1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, f'{name}.png'))
    plt.close(fig)


def plot_comparison(dir_path, scratch_curves, tl_curves, model_type):
    scratch_train_loss, scratch_train_acc, scratch_val_loss, scratch_val_acc = scratch_curves
    tl_train_loss, tl_train_acc, tl_val_loss, tl_val_acc = tl_curves

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_type} — Scratch vs Transfer Learning (best run)', fontsize=14)

    axes[0][0].plot(scratch_train_loss, label='Scratch', color='steelblue')
    axes[0][0].plot(tl_train_loss, label='TL', color='darkorange')
    axes[0][0].set_title('Train Loss')
    axes[0][0].set_xlabel('Epochs')
    axes[0][0].grid(True)
    axes[0][0].legend()

    axes[0][1].plot(scratch_val_loss, label='Scratch', color='steelblue')
    axes[0][1].plot(tl_val_loss, label='TL', color='darkorange')
    axes[0][1].set_title('Validation Loss')
    axes[0][1].set_xlabel('Epochs')
    axes[0][1].grid(True)
    axes[0][1].legend()

    axes[1][0].plot(scratch_train_acc, label='Scratch', color='steelblue')
    axes[1][0].plot(tl_train_acc, label='TL', color='darkorange')
    axes[1][0].set_title('Train Accuracy')
    axes[1][0].set_xlabel('Epochs')
    axes[1][0].grid(True)
    axes[1][0].legend()

    axes[1][1].plot(scratch_val_acc, label='Scratch', color='steelblue')
    axes[1][1].plot(tl_val_acc, label='TL', color='darkorange')
    axes[1][1].set_title('Validation Accuracy')
    axes[1][1].set_xlabel('Epochs')
    axes[1][1].grid(True)
    axes[1][1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, f'{model_type}_comparison.png'))
    plt.close(fig)


def plot_confusion(dir_path, cm, name='confusion'):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(dir_path, f'{name}.png'))
    plt.close()


def plot_predictions(dir_path, model, loader, class_names, device, name='predictions', n=10):
    # ImageNet normalisation parameters (to undo them for display)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    model.eval()
    images, labels = next(iter(loader))
    images_dev = images[:n].to(device)
    labels = labels[:n]

    with torch.no_grad():
        preds = model(images_dev).argmax(dim=1).cpu()

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle(name.replace('_', ' ').title(), fontsize=13)

    for i, ax in enumerate(axes.flat):
        img = images[i].cpu() * std + mean          # denormalise
        img = img.permute(1, 2, 0).clamp(0, 1)      # CHW → HWC, clip to [0,1]
        ax.imshow(img.numpy())
        color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_title(f"T: {class_names[labels[i]]}\nP: {class_names[preds[i]]}", color=color, fontsize=8)
        ax.axis('off')

    fig.tight_layout()
    fig.savefig(os.path.join(dir_path, f'{name}.png'))
    plt.close(fig)

