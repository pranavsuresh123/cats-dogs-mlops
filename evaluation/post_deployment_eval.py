import sys
from pathlib import Path

import requests
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def main():

    if len(sys.argv) != 2:
        print(
            "Usage: python evaluation/post_deployment_eval.py "
            "<base_url>"
        )
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")

    sample_dir = Path("evaluation/samples")

    y_true = []
    y_pred = []

    for image_path in sorted(sample_dir.glob("*.jpg")):

        filename = image_path.name.lower()

        if filename.startswith("cat_"):
            true_label = "cat"
        elif filename.startswith("dog_"):
            true_label = "dog"
        else:
            print(
                f"Skipping unknown file: {image_path.name}"
            )
            continue

        with image_path.open("rb") as image:

            response = requests.post(
                f"{base_url}/predict",
                files={
                    "file": (
                        image_path.name,
                        image,
                        "image/jpeg",
                    )
                },
                timeout=60,
            )

        response.raise_for_status()

        result = response.json()

        predicted_label = result["class"]

        y_true.append(true_label)
        y_pred.append(predicted_label)

        print(
            f"{image_path.name}: "
            f"true={true_label}, "
            f"predicted={predicted_label}, "
            f"confidence={result['probability']:.4f}"
        )

    if not y_true:
        print("No evaluation images found.")
        sys.exit(1)

    labels = ["cat", "dog"]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average="binary",
        pos_label="dog",
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average="binary",
        pos_label="dog",
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="binary",
        pos_label="dog",
        zero_division=0,
    )

    print("\nPost-Deployment Evaluation")
    print("==========================")
    print(f"Samples   : {len(y_true)}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")


if __name__ == "__main__":
    main()