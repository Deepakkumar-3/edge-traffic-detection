import time
from ultralytics import YOLO

IMAGE = "val1.jpg"
WARMUP_RUNS = 5
TEST_RUNS = 30

MODELS = {
    "PyTorch FP32": "models/best.pt",
    "OpenVINO FP": "models/best_openvino_model",
    "OpenVINO INT8": "models/best_int8_openvino_model",
}


def benchmark(model_path, model_name):
    print(f"\nBenchmarking {model_name}")
    print("-" * 45)

    model = YOLO(model_path)

    for _ in range(WARMUP_RUNS):
        model.predict(
            source=IMAGE,
            imgsz=640,
            rect=False,
            verbose=False
        )

    times = []

    for _ in range(TEST_RUNS):
        start = time.perf_counter()

        model.predict(
            source=IMAGE,
            imgsz=640,
            rect=False,
            verbose=False
        )

        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_latency = sum(times) / len(times)
    min_latency = min(times)
    max_latency = max(times)
    fps = 1000 / avg_latency

    print(f"Average latency : {avg_latency:.2f} ms")
    print(f"Minimum latency : {min_latency:.2f} ms")
    print(f"Maximum latency : {max_latency:.2f} ms")
    print(f"Approx FPS      : {fps:.2f}")

    return {
        "latency": avg_latency,
        "fps": fps
    }


results = {}

for name, path in MODELS.items():
    results[name] = benchmark(path, name)


print("\n" + "=" * 55)
print("FINAL COMPARISON")
print("=" * 55)

for name, result in results.items():
    print(
        f"{name:18s} | "
        f"{result['latency']:7.2f} ms | "
        f"{result['fps']:7.2f} FPS"
    )


fp32_latency = results["PyTorch FP32"]["latency"]
ov_latency = results["OpenVINO FP"]["latency"]
int8_latency = results["OpenVINO INT8"]["latency"]

print("\nSpeedups:")
print(f"OpenVINO FP vs PyTorch : {fp32_latency / ov_latency:.2f}x")
print(f"INT8 vs PyTorch        : {fp32_latency / int8_latency:.2f}x")
print(f"INT8 vs OpenVINO FP    : {ov_latency / int8_latency:.2f}x")