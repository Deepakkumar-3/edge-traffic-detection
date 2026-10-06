import cv2
import time
import os
from ultralytics import YOLO

MODEL_PATH = "models/best_int8_openvino_model"
SOURCE = "traffic.mp4"

OUTPUT_DIR = "outputs"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "traffic_detected.mp4")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(SOURCE)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read original video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Input video: {width}x{height}")
print(f"Original FPS: {original_fps:.2f}")

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    original_fps,
    (width, height)
)

frame_count = 0
total_latency = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video processing completed.")
        break

    start_time = time.perf_counter()

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000
    fps = 1000 / latency_ms

    total_latency += latency_ms
    frame_count += 1

    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Latency: {latency_ms:.1f} ms",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    writer.write(annotated_frame)

    cv2.imshow(
        "Edge Traffic Detection - OpenVINO INT8",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Stopped manually.")
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

if frame_count > 0:
    avg_latency = total_latency / frame_count
    avg_fps = 1000 / avg_latency

    print("\n" + "=" * 40)
    print("VIDEO SUMMARY")
    print("=" * 40)
    print(f"Frames processed : {frame_count}")
    print(f"Average latency  : {avg_latency:.2f} ms")
    print(f"Average FPS      : {avg_fps:.2f}")
    print(f"Saved output     : {OUTPUT_PATH}")