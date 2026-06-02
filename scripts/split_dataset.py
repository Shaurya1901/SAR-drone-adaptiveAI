import os, shutil, random
from pathlib import Path

IMG_DIR   = "data/processed/heridal/images"
LABEL_DIR = "data/processed/heridal/labels"

SPLITS = {
    "train": 0.70,
    "val":   0.20,
    "test":  0.10
}

FINAL_BASE = "data/final"

images = sorted(Path(IMG_DIR).glob("*.jpg"))

matched = []
missing_labels = []

for img_path in images:
    label_path = Path(LABEL_DIR) / (img_path.stem + ".txt")
    if label_path.exists():
        matched.append((img_path, label_path))
    else:
        missing_labels.append(img_path.name)

print(f"Total matched pairs : {len(matched)}")
print(f"Missing labels      : {len(missing_labels)}")
if missing_labels:
    print("  Examples:", missing_labels[:3])

random.seed(42)
random.shuffle(matched)


total     = len(matched)
n_train   = int(total * SPLITS["train"])
n_val     = int(total * SPLITS["val"])

n_test    = total - n_train - n_val

split_data = {
    "train": matched[:n_train],
    "val":   matched[n_train:n_train + n_val],
    "test":  matched[n_train + n_val:]
}

print(f"\nSplit sizes:")
print(f"  Train : {n_train}")
print(f"  Val   : {n_val}")
print(f"  Test  : {n_test}")
print(f"  Total : {n_train + n_val + n_test}")

for split, pairs in split_data.items():
    img_out   = Path(FINAL_BASE) / "images" / split
    label_out = Path(FINAL_BASE) / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_path in pairs:
        shutil.copy2(img_path,   img_out   / img_path.name)
        shutil.copy2(label_path, label_out / label_path.name)

    print(f"\n  [{split}] copied {len(pairs)} pairs")
    print(f"    → {img_out}")
    print(f"    → {label_out}")

print("\nDataset split complete.")