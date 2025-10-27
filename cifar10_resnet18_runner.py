# With augmentation
#python cifar10_resnet18_runner.py --gpu 0 --n_trials 10 --study_name "cifar10_resnet" --augment True

# Without augmentation
#python cifar10_resnet18_runner.py --gpu 1 --n_trials 10 --study_name "cifar10_resnet" --augment False

# Run parallel over 8 GPUs
# parallel --ungroup python cifar10_resnet18_runner.py --gpu {} --n_trials 10 --study_name "cifar10_resnet" --augment False ::: {0..7}

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Run Optuna optimization with Journal Storage for CIFAR-10 ResNet18.")
parser.add_argument("--gpu", type=int, default=-1, help="GPU device number (-1 for CPU)")
parser.add_argument("--n_trials", type=int, default=1, help="Number of trials to run")
parser.add_argument("--study_name", type=str, default="cifar10_study", help="Name of the Optuna study")
parser.add_argument("--augment", type=str, default="False", help="Use data augmentation (True/False)")
args = parser.parse_args()

# ----------------------------
# Set device
# ----------------------------
if args.gpu == -1 or not torch.cuda.is_available():
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(f"cuda:{args.gpu}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

print(f"Using device: {DEVICE}")

# ----------------------------
# Data setup
# ----------------------------
def get_dataloaders(batch_size, augment):
    if augment.lower() == "true":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader

# ----------------------------
# Training and evaluation
# ----------------------------
def train_one_epoch(model, optimizer, train_loader, device):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    return correct / total

# ----------------------------
# Objective function
# ----------------------------
def objective(trial):
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    T_max = trial.suggest_int("T_max", 20, 50)

    train_loader, val_loader = get_dataloaders(batch_size, args.augment)

    model = torchvision.models.resnet18(num_classes=10).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    max_epochs = 50
    for epoch in range(max_epochs):
        train_one_epoch(model, optimizer, train_loader, DEVICE)
        scheduler.step()
        val_accuracy = evaluate(model, val_loader, DEVICE)
        trial.report(val_accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_accuracy

# ----------------------------
# Run optimization
# ----------------------------
def run_optimization():
    storage = JournalStorage(JournalFileBackend(file_path="./journal.log"))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize"
    )
    study.optimize(objective, n_trials=args.n_trials)
    print(f"Study '{args.study_name}' completed. Best value: {study.best_value}, params: {study.best_params}")

if __name__ == "__main__":
    run_optimization()
