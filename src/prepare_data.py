from pathlib import Path
import random
import shutil

SEED = 42

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}


def get_images(directory: Path):
    return [
        file
        for file in directory.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_files(files, destination: Path):
    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in files:
        shutil.copy2(
            file,
            destination / file.name
        )


def split_class(class_name: str):

    source_dir = RAW_DIR / class_name

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {source_dir}"
        )

    images = get_images(source_dir)

    if not images:
        raise RuntimeError(
            f"No images found in {source_dir}"
        )

    random.seed(SEED)

    random.shuffle(images)

    total = len(images)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    train_images = images[:train_count]

    val_images = images[
        train_count:
        train_count + val_count
    ]

    test_images = images[
        train_count + val_count:
    ]

    copy_files(
        train_images,
        PROCESSED_DIR / "train" / class_name
    )

    copy_files(
        val_images,
        PROCESSED_DIR / "val" / class_name
    )

    copy_files(
        test_images,
        PROCESSED_DIR / "test" / class_name
    )

    print(
        f"{class_name}: "
        f"train={len(train_images)}, "
        f"val={len(val_images)}, "
        f"test={len(test_images)}"
    )


def main():

    if PROCESSED_DIR.exists():
        print("Removing existing processed dataset...")
        shutil.rmtree(PROCESSED_DIR)

    split_class("Cat")
    split_class("Dog")

    print("\nDataset split completed.")


if __name__ == "__main__":
    main()