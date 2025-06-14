import cv2
from ultralytics import YOLO
from collections import Counter

# Load YOLOv8 model
# model = YOLO("../yolov8n.pt")
model = YOLO("../yolov8-vietnamese-vehicle-30-epochs.pt")

# Find the first video file in the /videos directory
# (You can change this to your specific video file)
import os
video_files = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi', '.mov'))]
if not video_files:
    print("No video files found in the 'videos' directory.")
    exit()
video_path = os.path.join("videos", video_files[0])
print("Using video file:", video_path)

# Open video
cap = cv2.VideoCapture("videos/night_nguyentrai_ngatuso.MP4")  # Thay bằng tên video của bạn
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle']
all_detected = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on current frame
    results = model(frame)

    # Draw boxes and collect class names
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name in vehicle_classes:
                all_detected.append(class_name)

                # Draw bounding box + label
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                label = f"{class_name} {conf:.2f}"

                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0,255,0), 2)
                cv2.putText(frame, label, (xyxy[0], xyxy[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Show video frame
    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

# Count and display results
counts = Counter(all_detected)
print("\n🚗 Vehicle count in video:")
for cls in vehicle_classes:
    print(f"- {cls.capitalize():12}: {counts.get(cls, 0)}")
print(f"\n✅ Total vehicles detected: {sum(counts.values())}")
