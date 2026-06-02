import pandas as pd
import os
from pathlib import Path

CSV_PATH   = "data/raw/heridal/train/_annotations.csv"
IMG_SRC    = "data/raw/heridal/train"
LABEL_OUT  = "data/processed/heridal/labels"
IMG_OUT    = "data/processed/heridal/images"

os.makedirs(LABEL_OUT, exist_ok=True)
os.makedirs(IMG_OUT,   exist_ok=True)

df = pd.read_csv(CSV_PATH)
print(f"Total annotations: {len(df)}")
print(f"Unique images:     {df['filename'].nunique()}")

# ── Convert to YOLO format ─────────────────────────────────────────────
# YOLO format: class x_center y_center width height  (all normalized 0-1)
# class 0 = human

skipped = 0
processed = 0

for filename, group in df.groupby("filename"):

    # image dimensions (same for all rows of this image)
    img_w = group["width"].iloc[0]
    img_h = group["height"].iloc[0]

    lines = []
    for _, row in group.iterrows():
        xmin, ymin, xmax, ymax = row["xmin"], row["ymin"], row["xmax"], row["ymax"]

        # sanity check
        if xmax <= xmin or ymax <= ymin:
            skipped += 1
            continue

        # normalize
        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width    = (xmax - xmin) / img_w
        height   = (ymax - ymin) / img_h

        # clamp to [0,1] just in case
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width    = max(0, min(1, width))
        height   = max(0, min(1, height))

        lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    if not lines:
        skipped += 1
        continue

    # write label file (same name as image, .txt extension)
    stem = Path(filename).stem
    label_file = os.path.join(LABEL_OUT, stem + ".txt")
    with open(label_file, "w") as f:
        f.write("\n".join(lines))

    # copy image to processed folder
    src_img  = os.path.join(IMG_SRC,  filename)
    dst_img  = os.path.join(IMG_OUT,  filename)
    if os.path.exists(src_img):
        import shutil
        shutil.copy2(src_img, dst_img)
        processed += 1
    else:
        print(f"  WARNING: image not found → {filename}")

print(f"\nDone. Processed: {processed} images | Skipped: {skipped} annotations")