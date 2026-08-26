from pathlib import Path

import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.model import CatsDogsCNN
from src.preprocess import get_train_transform, get_eval_transform


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
ARTIFACT_DIR = Path("mlflow_artifacts")

BATCH_SIZE = 32
EPOCHS = 5                    # Start with 1 epoch as a test
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            predictions.extend(
                preds.cpu().numpy()
            )

            actuals.extend(
                labels.cpu().numpy()
            )

    average_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        actuals,
        predictions
    )

    return (
        average_loss,
        accuracy,
        actuals,
        predictions,
    )


# ============================================================
# Plots
# ============================================================

def save_loss_plot(train_losses, val_losses):

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure()

    plt.plot(
        train_losses,
        label="Train Loss"
    )

    plt.plot(
        val_losses,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.savefig(
        ARTIFACT_DIR / "loss_curve.png",
        bbox_inches="tight"
    )

    plt.close()


def save_accuracy_plot(
    train_accuracies,
    val_accuracies
):

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure()

    plt.plot(
        train_accuracies,
        label="Train Accuracy"
    )

    plt.plot(
        val_accuracies,
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.savefig(
        ARTIFACT_DIR / "accuracy_curve.png",
        bbox_inches="tight"
    )

    plt.close()


def save_confusion_matrix(
    actuals,
    predictions
):

    cm = confusion_matrix(
        actuals,
        predictions
    )

    plt.figure()

    plt.imshow(cm)

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.xticks(
        [0, 1],
        ["Cat", "Dog"]
    )

    plt.yticks(
        [0, 1],
        ["Cat", "Dog"]
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.savefig(
        ARTIFACT_DIR / "confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Main training function
# ============================================================

def main():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = ImageFolder(
        DATA_DIR / "train",
        transform=get_train_transform()
    )

    val_dataset = ImageFolder(
        DATA_DIR / "val",
        transform=get_eval_transform()
    )

    test_dataset = ImageFolder(
        DATA_DIR / "test",
        transform=get_eval_transform()
    )

    print(
        "Classes:",
        train_dataset.class_to_idx
    )

    print(
        f"Train images: {len(train_dataset)}"
    )

    print(
        f"Validation images: {len(val_dataset)}"
    )

    print(
        f"Test images: {len(test_dataset)}"
    )

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = CatsDogsCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_experiment(
        "Cats-Dogs-Classification"
    )

    with mlflow.start_run():

        mlflow.log_param(
            "model",
            "CatsDogsCNN"
        )

        mlflow.log_param(
            "image_size",
            224
        )

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE
        )

        mlflow.log_param(
            "epochs",
            EPOCHS
        )

        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE
        )

        mlflow.log_param(
            "optimizer",
            "Adam"
        )

        mlflow.log_param(
            "device",
            str(DEVICE)
        )

        train_losses = []
        val_losses = []

        train_accuracies = []
        val_accuracies = []

        best_val_accuracy = 0.0

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for epoch in range(EPOCHS):

            model.train()

            running_loss = 0.0

            train_predictions = []
            train_actuals = []

            for batch_idx, (
                images,
                labels
            ) in enumerate(train_loader):

                images = images.to(
                    DEVICE,
                    non_blocking=True
                )

                labels = labels.to(
                    DEVICE,
                    non_blocking=True
                )

                optimizer.zero_grad()

                outputs = model(images)

                loss = criterion(
                    outputs,
                    labels
                )

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                predictions = outputs.argmax(
                    dim=1
                )

                train_predictions.extend(
                    predictions.detach()
                    .cpu()
                    .numpy()
                )

                train_actuals.extend(
                    labels.detach()
                    .cpu()
                    .numpy()
                )

            train_loss = (
                running_loss /
                len(train_loader)
            )

            train_accuracy = accuracy_score(
                train_actuals,
                train_predictions
            )

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            (
                val_loss,
                val_accuracy,
                _,
                _
            ) = evaluate(
                model,
                val_loader,
                criterion
            )

            train_losses.append(
                train_loss
            )

            val_losses.append(
                val_loss
            )

            train_accuracies.append(
                train_accuracy
            )

            val_accuracies.append(
                val_accuracy
            )

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

            mlflow.log_metric(
                "train_loss",
                train_loss,
                step=epoch
            )

            mlflow.log_metric(
                "train_accuracy",
                train_accuracy,
                step=epoch
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch
            )

            mlflow.log_metric(
                "val_accuracy",
                val_accuracy,
                step=epoch
            )

            # Save best model
            if val_accuracy > best_val_accuracy:

                best_val_accuracy = val_accuracy

                torch.save(
                    model.state_dict(),
                    MODEL_DIR / "model.pt"
                )

        # ----------------------------------------------------
        # Test
        # ----------------------------------------------------

        # Make sure some model was saved
        if not (MODEL_DIR / "model.pt").exists():

            torch.save(
                model.state_dict(),
                MODEL_DIR / "model.pt"
            )

        model.load_state_dict(
            torch.load(
                MODEL_DIR / "model.pt",
                map_location=DEVICE
            )
        )

        (
            test_loss,
            test_accuracy,
            actuals,
            predictions
        ) = evaluate(
            model,
            test_loader,
            criterion
        )

        precision = precision_score(
            actuals,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            actuals,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            actuals,
            predictions,
            zero_division=0
        )

        print("\nTest Results")
        print("============")
        print(
            f"Test Loss : {test_loss:.4f}"
        )

        print(
            f"Accuracy  : {test_accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        # ----------------------------------------------------
        # MLflow test metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "test_loss",
            test_loss
        )

        mlflow.log_metric(
            "test_accuracy",
            test_accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # ----------------------------------------------------
        # Save artifacts
        # ----------------------------------------------------

        save_loss_plot(
            train_losses,
            val_losses
        )

        save_accuracy_plot(
            train_accuracies,
            val_accuracies
        )

        save_confusion_matrix(
            actuals,
            predictions
        )

        mlflow.log_artifact(
            ARTIFACT_DIR / "loss_curve.png"
        )

        mlflow.log_artifact(
            ARTIFACT_DIR / "accuracy_curve.png"
        )

        mlflow.log_artifact(
            ARTIFACT_DIR / "confusion_matrix.png"
        )

        mlflow.log_artifact(
            MODEL_DIR / "model.pt"
        )

        print("\nTraining completed successfully.")
        print(
            f"Model saved to: {MODEL_DIR / 'model.pt'}"
        )


if __name__ == "__main__":
    main()