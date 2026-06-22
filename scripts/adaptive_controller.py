"""
Adaptive Inference Controller — SAR Drone
Dynamically selects model tier based on real-time SOC discharge rate,
which itself depends on fluctuating propulsion power (simulating real flight
conditions) and the power draw of the currently active inference tier.
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
import random

# ── Configuration ─────────────────────────────────────────────────────

MODELS = {
    "Tier 1": {"path": "models/tier1_fp32/tier1_fp32_640.tflite",  "imgsz": 640, "power_W": 5.5},
    "Tier 2": {"path": "models/tier2_int8/tier2_int8_640.tflite",  "imgsz": 640, "power_W": 4.5},
    "Tier 3": {"path": "models/tier3_int8_320/tier3_int8_320.tflite", "imgsz": 320, "power_W": 3.5},
}

BATTERY_Wh        = 77.0      # rated battery capacity
PROPULSION_BASE_W = 150.0     # cruise propulsion baseline
NOISE_STD_W       = 10.0      # turbulence/wind fluctuation
SPIKE_PROB        = 0.05      # probability of a maneuver spike per frame
SPIKE_MAGNITUDE_W = 30.0      # extra power during a maneuver spike

T_HIGH_MARGIN_MIN = 20.0      # switch to Tier 2 when Tier-1 projected time < this
T_LOW_MARGIN_MIN  = 5.0       # switch to Tier 3 when Tier-2 projected time < this

TEST_IMAGES_DIR   = "data/final/images/test"
LOG_OUTPUT        = "results/logs/adaptive_run.csv"

random.seed(42)
np.random.seed(42)

# ── All source folders (no copying — read directly) ────────────────────
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

def gather_all_images():
    """Collect every image path across all sources, then shuffle once."""
    all_images = []
    for folder in IMAGE_SOURCES:
        p = Path(folder)
        if not p.exists():
            print(f"  WARNING: path not found → {folder}")
            continue
        imgs = list(p.glob("*.jpg")) + list(p.glob("*.png")) + list(p.glob("*.JPG")) + list(p.glob("*.PNG"))
        all_images.extend(imgs)
        print(f"  {folder} → {len(imgs)} images")
    random.shuffle(all_images)
    return all_images

# ── Load all 3 interpreters once ─────────────────────────────────────

def load_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path, num_threads=1)
    interp.allocate_tensors()
    return interp

print("Loading all 3 tier models...")
interpreters = {}
for tier, cfg in MODELS.items():
    interpreters[tier] = load_interpreter(cfg["path"])
    print(f"  {tier} loaded ({cfg['imgsz']}px, {cfg['power_W']}W)")

# ── Propulsion power simulator ──────────────────────────────────────

def get_propulsion_power():
    """Simulates fluctuating propulsion power — wind, maneuvers, etc."""
    noise = np.random.normal(0, NOISE_STD_W)
    spike = SPIKE_MAGNITUDE_W if random.random() < SPIKE_PROB else 0.0
    return max(50.0, PROPULSION_BASE_W + noise + spike)  # floor at 50W

# ── SOC discharge model ──────────────────────────────────────────────

def discharge_rate_pct_per_sec(p_total_W, battery_Wh):
    """dSoC/dt — percentage points lost per second at this power draw"""
    return (p_total_W / (battery_Wh * 3600)) * 100

def projected_remaining_min(soc_pct, p_total_W, battery_Wh):
    """How many minutes left if this power draw continued"""
    rate_pct_per_sec = discharge_rate_pct_per_sec(p_total_W, battery_Wh)
    if rate_pct_per_sec <= 0:
        return float('inf')
    seconds_remaining = soc_pct / rate_pct_per_sec
    return seconds_remaining / 60.0

# ── Tier selection logic ──────────────────────────────────────────────

def select_tier(soc_pct, propulsion_W):
    """
    Projects remaining flight time under each tier's power draw,
    picks the lightest tier that still satisfies the time margins.
    """
    # Try Tier 1 first
    p_total_t1 = propulsion_W + MODELS["Tier 1"]["power_W"]
    time_t1 = projected_remaining_min(soc_pct, p_total_t1, BATTERY_Wh)

    if time_t1 >= T_HIGH_MARGIN_MIN:
        return "Tier 1", time_t1

    # Try Tier 2
    p_total_t2 = propulsion_W + MODELS["Tier 2"]["power_W"]
    time_t2 = projected_remaining_min(soc_pct, p_total_t2, BATTERY_Wh)

    if time_t2 >= T_LOW_MARGIN_MIN:
        return "Tier 2", time_t2

    # Fall back to Tier 3
    p_total_t3 = propulsion_W + MODELS["Tier 3"]["power_W"]
    time_t3 = projected_remaining_min(soc_pct, p_total_t3, BATTERY_Wh)
    return "Tier 3", time_t3

# ── Inference helper ────────────────────────────────────────────────

def run_inference(tier, image):
    interp = interpreters[tier]
    input_details  = interp.get_input_details()
    output_details = interp.get_output_details()
    imgsz = MODELS[tier]["imgsz"]

    # Resize + normalize dummy/real image to match tier's expected input
    img_resized = tf.image.resize(image, (imgsz, imgsz)).numpy()
    img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0

    start = time.perf_counter()
    interp.set_tensor(input_details[0]['index'], img_resized)
    interp.invoke()
    output = interp.get_tensor(output_details[0]['index'])
    latency_ms = (time.perf_counter() - start) * 1000

    # Count detections above confidence 0.25 (YOLO output: [1, 5, N])
    # output[0] shape = (5, num_boxes) -> [x, y, w, h, conf]
    confidences = output[0][4]
    detections = int(np.sum(confidences > 0.25))

    return latency_ms, detections

# ── Main simulation loop ──────────────────────────────────────────────

def main():
    image_paths = gather_all_images()
    print(f"\nTotal images available across all datasets: {len(image_paths)}")
    print(f"Running adaptive controller until battery depletes...\n")

    soc = 100.0
    logs = []
    frame_idx = 0
    MAX_FRAMES = len(image_paths)  # cap at total available images, no repeats needed now

    while soc > 0 and frame_idx < MAX_FRAMES:
        img_path = image_paths[frame_idx]  # no modulo needed — full pool, no repeats

        propulsion_W = get_propulsion_power()
        tier, projected_min = select_tier(soc, propulsion_W)

        img = tf.io.read_file(str(img_path))
        img = tf.image.decode_jpeg(img, channels=3)
        latency_ms, detections = run_inference(tier, img)

        inference_power_W = MODELS[tier]["power_W"]
        total_power_W = propulsion_W + inference_power_W
        energy_J = inference_power_W * (latency_ms / 1000.0)

        dt_sec = latency_ms / 1000.0
        soc_drop = discharge_rate_pct_per_sec(total_power_W, BATTERY_Wh) * dt_sec
        soc = max(0.0, soc - soc_drop)

        logs.append({
            "frame": frame_idx,
            "image": img_path.name,
            "source_dataset": img_path.parts[-3] if "raw" in str(img_path) else "heridal",
            "soc_pct": round(soc, 3),
            "propulsion_W": round(propulsion_W, 2),
            "tier": tier,
            "inference_power_W": inference_power_W,
            "total_power_W": round(total_power_W, 2),
            "latency_ms": round(latency_ms, 2),
            "compute_energy_J": round(energy_J, 4),
            "projected_remaining_min": round(projected_min, 2),
            "detections": detections,
        })

        if frame_idx % 200 == 0:
            print(f"Frame {frame_idx:5d} | SOC: {soc:6.2f}% | Propulsion: {propulsion_W:6.1f}W | "
                  f"Tier: {tier} | Latency: {latency_ms:6.1f}ms | Detections: {detections}")

        frame_idx += 1

    if soc <= 0:
        print(f"\nBattery depleted at frame {frame_idx} out of {MAX_FRAMES} available images.")
    else:
        print(f"\nRan through all {MAX_FRAMES} available images. Final SOC: {soc:.2f}% (battery not depleted)")

    df = pd.DataFrame(logs)
    Path(LOG_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LOG_OUTPUT, index=False)

    print(f"\nSimulation complete. {len(logs)} images processed before stopping.")
    print(f"Log saved to {LOG_OUTPUT}")

    print("\n" + "="*60)
    print("TIER USAGE SUMMARY")
    print("="*60)
    tier_counts = df["tier"].value_counts()
    for tier, count in tier_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {tier}: {count} frames ({pct:.1f}%)")

    print("\n" + "="*60)
    print("SOURCE DATASET BREAKDOWN")
    print("="*60)
    source_counts = df["source_dataset"].value_counts()
    for src, count in source_counts.items():
        print(f"  {src}: {count} images")

    total_compute_energy = df["compute_energy_J"].sum()
    print(f"\nTotal compute energy used: {total_compute_energy:.2f} J")

if __name__ == "__main__":
    main()