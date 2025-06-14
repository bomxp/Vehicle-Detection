from ultralytics import YOLO

# Load model (bạn có thể chọn yolov8n.pt, yolov8s.pt, yolov8m.pt,...)
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="vietnamese-vehicle-3/data.yaml",
    epochs=20, # Số epochs có thể điều chỉnh tùy theo độ lớn của dataset
    patience=5, # Số epochs và patience có thể điều chỉnh, sau 5 epochs nếu không cải thiện thì dừng
    imgsz=640, # Kích thước ảnh đầu vào, có thể là 640, 1280, ...
    batch=16, # Kích thước batch, có thể điều chỉnh tùy theo GPU của bạn
    project="vehicle-detection-project",
    name="yolov8-vietnamese-vehicle-20",
    pretrained=True # Sử dụng mô hình đã được huấn luyện trước (pretrained model)
)
