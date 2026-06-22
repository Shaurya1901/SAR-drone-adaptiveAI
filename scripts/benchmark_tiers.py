import tensorflow as tf
import numpy as np
import time
import json
from pathlib import Path

MODELS = {
    "Tier 1 FP32 640": ("models/tier1_fp32/tier1_fp32_640.tflite", 640),
    "Tier 2 INT8 640": ("models/tier2_int8/tier2_int8_640.tflite", 640),
    "Tier 3 INT8 320": ("models/tier3_int8_320/tier3_int8_320.tflite", 320),
}

NUM_WARMUP = 5  
NUM_RUNS   = 30

results = {}

for tier_name, (model_path, imgsz) in MODELS.items():
    print(f"\nBenchmarking {tier_name}...")
    print(f"  Model: {model_path}")

    interpreter = tf.lite.Interpreter(
        model_path   = model_path,
        num_threads  = 4
    )
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape    = input_details[0]['shape']
    input_dtype    = input_details[0]['dtype']

    print(f"  Input shape : {input_shape}")
    print(f"  Input dtype : {input_dtype}")

    def make_dummy_input():
        img = np.random.randint(0, 255, (1, imgsz, imgsz, 3), dtype=np.uint8)
        if input_dtype == np.float32:
            img = img.astype(np.float32) / 255.0
        elif input_dtype == np.int8:
            img = img.astype(np.int8)
        return img

    print(f"  Warming up ({NUM_WARMUP} runs)...")
    for _ in range(NUM_WARMUP):
        interpreter.set_tensor(input_details[0]['index'], make_dummy_input())
        interpreter.invoke()

    print(f"  Measuring ({NUM_RUNS} runs)...")
    latencies = []
    for _ in range(NUM_RUNS):
        img = make_dummy_input()
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    mean_ms   = float(np.mean(latencies))
    std_ms    = float(np.std(latencies))
    min_ms    = float(np.min(latencies))
    max_ms    = float(np.max(latencies))

    results[tier_name] = {
        "mean_ms" : round(mean_ms, 2),
        "std_ms"  : round(std_ms,  2),
        "min_ms"  : round(min_ms,  2),
        "max_ms"  : round(max_ms,  2),
    }

    print(f"  Mean  : {mean_ms:.1f} ms")
    print(f"  Std   : {std_ms:.1f} ms")
    print(f"  Min   : {min_ms:.1f} ms")
    print(f"  Max   : {max_ms:.1f} ms")

out_path = Path("results/logs/benchmark_local.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n\nBenchmark saved to {out_path}")
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
for tier, vals in results.items():
    print(f"{tier:25s} → {vals['mean_ms']:7.1f} ms ± {vals['std_ms']:.1f} ms")
print("="*55)