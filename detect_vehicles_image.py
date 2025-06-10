from ultralytics import YOLO
from collections import Counter

# Load model
model = YOLO("yolov8n.pt")  # hoặc yolov8s.pt nếu muốn chính xác hơn

# Detect objects in image
results = model("images/traffic2.webp", show=True)

# delay 5s
import time
time.sleep(3)

# Extract class names
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
detected_classes = []

for r in results:
    for box in r.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Only count known vehicle classes
        if class_name in vehicle_classes:
            detected_classes.append(class_name)

# Count each vehicle type
counts = Counter(detected_classes)

# Display result
print("\n🚗 Vehicle count by type:")
for cls in vehicle_classes:
    print(f"- {cls.capitalize():12}: {counts.get(cls, 0)}")

print(f"\n✅ Total vehicles: {sum(counts.values())}")
