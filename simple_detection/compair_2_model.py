from ultralytics import YOLO
import cv2

# Model paths and image path
model_path1 = "../yolov8n.pt"
model_path2 = "../yolov8-vietnamese-vehicle-20.pt"
# image_path = "images/traffic2.webp"
image_path = "images/night_nguyentrai_ngatuso1.jpg"

# Load models
model1 = YOLO(model_path1)
model2 = YOLO(model_path2)

# Detect objects
results1 = model1(image_path)
results2 = model2(image_path)

# Read image twice for separate drawing
img1 = cv2.imread(image_path)
img2 = cv2.imread(image_path)

# Draw detections for model1
for r in results1:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = model1.names[class_id]
        cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img1, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Draw detections for model2
for r in results2:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = model2.names[class_id]
        cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img2, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Concatenate images horizontally
combined = cv2.hconcat([img1, img2])

# Show combined image with both model names
window_title = f"YOLO Models: {model_path1} ================================================== {model_path2}"
cv2.imshow(window_title, combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
