import os
import sys
import torch
import numpy as np
import time
from tabulate import tabulate

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))

from dataset import get_loaders
from model import Model, Architecture
from training import train, get_metrics
from predict import predict_samples
import argparse
import draw

CLASS_NAMES = [
    'apple_pie', 'caesar_salad', 'clam_chowder', 'edamame',
    'french_fries', 'hamburger', 'hot_dog', 'ice_cream', 'sushi', 'waffles'
]

def parse_args():
    parser = argparse.ArgumentParser(description="Food101 CNN: Scratch vs Transfer Learning")
    parser.add_argument("--model_type", type=str,
                        choices=['alexnet', 'resnet34', 'mobilenet_v3_large'],
                        default='resnet34')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("-o", "--output_path", type=str, default="results",
                        help="Path to save results")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'],
                        default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_experiment(model_type, pretrained, seeds, args, device, base_output_path):
    mode_name = "tl" if pretrained else "scratch"

    all_test_accs = []
    all_run_curves = []

    best_val_acc = 0.0
    best_model = None
    best_test_loader = None
    best_curves = None
    best_conf_matrix = None
    best_sensitivity = None
    best_specificity = None

    for seed in seeds:
        print(f"\n=== {model_type.upper()} | {'TL' if pretrained else 'Scratch'} | Seed {seed} ===")

        train_loader, val_loader, test_loader = get_loaders(
            batch_size=args.batch_size,
            seed=seed,
            augment=args.augment
        )

        arch = Architecture(input_dim=3, output_dim=10)
        model = Model(arch, model_type=model_type, pretrained=pretrained).to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        log_dir = os.path.join(base_output_path, "logs", f"{model_type}_{mode_name}_seed{seed}")

        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for t_loss, t_acc, v_loss, v_acc in train(
            model, train_loader, val_loader, criterion, optimizer, device,
            num_epochs=args.epochs, log_dir=log_dir
        ):
            train_losses.append(t_loss)
            val_losses.append(v_loss)
            train_accs.append(t_acc)
            val_accs.append(v_acc)

        test_acc, conf_matrix, sensitivity, specificity = get_metrics(model, test_loader, device)
        print(f"Test Acc: {test_acc:.4f}")

        all_test_accs.append(test_acc)
        all_run_curves.append((train_losses, train_accs, val_losses, val_accs))

        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_model = model
            best_test_loader = test_loader
            best_curves = (train_losses, train_accs, val_losses, val_accs)
            best_conf_matrix = conf_matrix
            best_sensitivity = sensitivity
            best_specificity = specificity

    mean_acc = float(np.mean(all_test_accs))
    std_acc = float(np.std(all_test_accs))

    return {
        "model": best_model,
        "test_loader": best_test_loader,
        "best_curves": best_curves,
        "all_curves": all_run_curves,
        "test_accs": all_test_accs,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "conf_matrix": best_conf_matrix,
        "sensitivity": best_sensitivity,
        "specificity": best_specificity,
    }


def compare(args, device):
    output_path = os.path.join(os.path.dirname(__file__), 'output', args.output_path, args.model_type)
    os.makedirs(output_path, exist_ok=True)

    seeds = [1, 2, 3]

    print(f"\n{'='*50}\nFROM SCRATCH — {args.model_type}\n{'='*50}")
    scratch = run_experiment(args.model_type, pretrained=False, seeds=seeds,
                             args=args, device=device, base_output_path=output_path)

    print(f"\n{'='*50}\nTRANSFER LEARNING — {args.model_type}\n{'='*50}")
    tl = run_experiment(args.model_type, pretrained=True, seeds=seeds,
                        args=args, device=device, base_output_path=output_path)

    # --- Text results ---
    output_file = os.path.join(output_path, "result.txt")
    with open(output_file, "w") as f:
        f.write("=== Training Configuration ===\n")
        headers = ["Device", "Model", "Epochs", "LR", "Batch Size", "Augment"]
        rows = [[str(device), args.model_type, args.epochs, args.lr, args.batch_size, args.augment]]
        f.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

        f.write("\n=== Test Accuracy (3 runs) ===\n")
        headers = ["Mode", "Run 1", "Run 2", "Run 3", "Mean", "Std"]
        rows = [
            ["Scratch"] + [f"{a:.4f}" for a in scratch["test_accs"]] + [f"{scratch['mean_acc']:.4f}", f"{scratch['std_acc']:.4f}"],
            ["TL"]      + [f"{a:.4f}" for a in tl["test_accs"]]      + [f"{tl['mean_acc']:.4f}",      f"{tl['std_acc']:.4f}"],
        ]
        f.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

        for mode_name, result in [("Scratch", scratch), ("TL", tl)]:
            f.write(f"\n=== {mode_name} — Per-class Metrics (best run) ===\n")
            headers = ["Class", "Sensitivity", "Specificity"]
            rows = [[CLASS_NAMES[i], f"{result['sensitivity'][i]:.4f}", f"{result['specificity'][i]:.4f}"]
                    for i in range(len(CLASS_NAMES))]
            f.write(tabulate(rows, headers=headers, tablefmt="grid") + "\n")

    print(f"\nScratch: {scratch['mean_acc']:.4f} ± {scratch['std_acc']:.4f}")
    print(f"TL:      {tl['mean_acc']:.4f} ± {tl['std_acc']:.4f}")

    # --- Plots ---
    draw.plot_metrics(output_path, *scratch["best_curves"],
                      name=f"{args.model_type}_scratch_best")
    draw.plot_metrics(output_path, *tl["best_curves"],
                      name=f"{args.model_type}_tl_best")
    draw.plot_comparison(output_path, scratch["best_curves"], tl["best_curves"], args.model_type)
    draw.plot_confusion(output_path, scratch["conf_matrix"],
                        name=f"{args.model_type}_scratch_confusion")
    draw.plot_confusion(output_path, tl["conf_matrix"],
                        name=f"{args.model_type}_tl_confusion")
    predict_samples(scratch["model"], device, output_path,
                    name=f"{args.model_type}_scratch_predictions")
    predict_samples(tl["model"], device, output_path,
                    name=f"{args.model_type}_tl_predictions")


def main():
    args = parse_args()
    device = torch.device(args.device)
    compare(args, device)


if __name__ == "__main__":
    main()
