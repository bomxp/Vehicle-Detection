from ultralytics import YOLO

# Load model (bạn có thể chọn yolov8n.pt, yolov8s.pt, yolov8m.pt,...)
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="vietnamese-vehicle-3/data.yaml",
    epochs=50, # Số epochs có thể điều chỉnh tùy theo độ lớn của dataset
    patience=10, # Số epochs và patience có thể điều chỉnh, sau 5 epochs nếu không cải thiện thì dừng
    # imgsz=640, # Kích thước ảnh đầu vào, có thể là 640, 1280, ...
    batch=16, # Kích thước batch, có thể điều chỉnh tùy theo GPU của bạn
    freeze=15,  # Freeze 15 layer đầu (tức phần backbone)
    project="vehicle-detection-project",
    name="yolov8n-vietnamese-vehicle-50-epochs-freeze-15",
    pretrained=True # Sử dụng mô hình đã được huấn luyện trước (pretrained model)
)
