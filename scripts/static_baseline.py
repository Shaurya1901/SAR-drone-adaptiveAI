"""
Static Baseline Controller — Tier 1 Only
Runs the SAME image pool through Tier 1 exclusively, no adaptive switching,
for direct comparison against the adaptive controller's energy consumption.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
import random

# ── Configuration (same as adaptive controller) ────────────────────────

MODEL_PATH = "models/tier1_fp32/tier1_fp32_640.tflite"
MODEL_POWER_W = 5.5
MODEL_IMGSZ = 640

BATTERY_Wh        = 77.0
PROPULSION_BASE_W = 150.0
NOISE_STD_W       = 10.0
SPIKE_PROB        = 0.05
SPIKE_MAGNITUDE_W = 30.0

IMAGE_SOURCES = [
    "data/processed/heridal/images",
    "data/raw/visdrone/VisDrone2019-DET-train/VisDrone2019-DET-train/images",
    "data/raw/visdrone/VisDrone2019-DET-val/VisDrone2019-DET-val/images",
    "data/raw/visdrone/VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev/images",
    "data/raw/sard/search-and-rescue/train/images",
    "data/raw/sard/search-and-rescue/valid/images",
    "data/raw/sard/search-and-rescue/test/images",
    "data/raw/c2a/C2A_Dataset/new_dataset3/train/images",
    "data/raw/c2a/C2A_Dataset/new_dataset3/val/images",
    "data/raw/c2a/C2A_Dataset/new_dataset3/test/images",
]

LOG_OUTPUT = "results/logs/static_baseline_run.csv"

random.seed(42)   # SAME seed as adaptive run — ensures identical image order
np.random.seed(42)

# ── Load Tier 1 only ─────────────────────────────────────────────────

def load_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    return interp

print("Loading Tier 1 model (static baseline)...")
interpreter = load_interpreter(MODEL_PATH)
print(f"  Tier 1 loaded ({MODEL_IMGSZ}px, {MODEL_POWER_W}W)")

# ── Propulsion simulator (identical to adaptive script) ────────────────

def get_propulsion_power():
    noise = np.random.normal(0, NOISE_STD_W)
    spike = SPIKE_MAGNITUDE_W if random.random() < SPIKE_PROB else 0.0
    return max(50.0, PROPULSION_BASE_W + noise + spike)

def discharge_rate_pct_per_sec(p_total_W, battery_Wh):
    return (p_total_W / (battery_Wh * 3600)) * 100

# ── Inference helper (identical logic) ──────────────────────────────

def run_inference(image):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img_resized = tf.image.resize(image, (MODEL_IMGSZ, MODEL_IMGSZ)).numpy()
    img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    start = time.perf_counter()
    interpreter.set_tensor(input_details[0]['index'], img_resized)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    latency_ms = (time.perf_counter() - start) * 1000

    confidences = output[0][4]
    detections = int(np.sum(confidences > 0.25))

    return latency_ms, detections

# ── Gather images — SAME pooling + shuffle logic as adaptive run ───────

def gather_all_images():
    all_images = []
    for folder in IMAGE_SOURCES:
        p = Path(folder)
        if not p.exists():
            print(f"  WARNING: path not found → {folder}")
            continue
        imgs = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.JPG")) + list(p.glob("*.PNG"))
        all_images.extend(imgs)
    random.shuffle(all_images)
    return all_images

# ── Main loop ───────────────────────────────────────────────────────

def main():
    image_paths = gather_all_images()
    print(f"\nTotal images available: {len(image_paths)}")
    print(f"Running STATIC Tier 1 baseline until battery depletes...\n")

    soc = 100.0
    logs = []
    frame_idx = 0
    MAX_FRAMES = len(image_paths)

    while soc > 0 and frame_idx < MAX_FRAMES:
        img_path = image_paths[frame_idx]

        propulsion_W = get_propulsion_power()

        img = tf.io.read_file(str(img_path))
        img = tf.image.decode_jpeg(img, channels=3)
        latency_ms, detections = run_inference(img)

        total_power_W = propulsion_W + MODEL_POWER_W
        energy_J = MODEL_POWER_W * (latency_ms / 1000.0)

        dt_sec = latency_ms / 1000.0
        soc_drop = discharge_rate_pct_per_sec(total_power_W, BATTERY_Wh) * dt_sec
        soc = max(0.0, soc - soc_drop)

        logs.append({
            "frame": frame_idx,
            "image": img_path.name,
            "source_dataset": img_path.parts[-3] if "raw" in str(img_path) else "heridal",
            "soc_pct": round(soc, 3),
            "propulsion_W": round(propulsion_W, 2),
            "tier": "Tier 1 (static)",
            "inference_power_W": MODEL_POWER_W,
            "total_power_W": round(total_power_W, 2),
            "latency_ms": round(latency_ms, 2),
            "compute_energy_J": round(energy_J, 4),
            "detections": detections,
        })

        if frame_idx % 200 == 0:
            print(f"Frame {frame_idx:5d} | SOC: {soc:6.2f}% | Propulsion: {propulsion_W:6.1f}W | "
                  f"Latency: {latency_ms:6.1f}ms | Detections: {detections}")

        frame_idx += 1

    if soc <= 0:
        print(f"\nBattery depleted at frame {frame_idx} out of {MAX_FRAMES} available images.")
    else:
        print(f"\nRan through all {MAX_FRAMES} images. Final SOC: {soc:.2f}% (battery not depleted)")

    df = pd.DataFrame(logs)
    Path(LOG_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_OUTPUT, index=False)

    print(f"\nStatic baseline simulation complete. {len(logs)} images processed.")
    print(f"Log saved to {LOG_OUTPUT}")

    total_compute_energy = df["compute_energy_J"].sum()
    total_detections = df["detections"].sum()

    print("\n" + "="*60)
    print("STATIC BASELINE SUMMARY")
    print("="*60)
    print(f"  Frames processed     : {len(df)}")
    print(f"  Total compute energy : {total_compute_energy:.2f} J")
    print(f"  Total detections     : {total_detections}")
    print(f"  Avg latency          : {df['latency_ms'].mean():.1f} ms")

if __name__ == "__main__":
    main()