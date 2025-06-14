from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")  # 👈 Thay bằng API key của bạn
project = rf.workspace("car-classification").project("vietnamese-vehicle")
dataset = project.version(3).download("yolov8")  # hoặc "yolov5"
