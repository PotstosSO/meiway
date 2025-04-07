from ultralytics import YOLO

# Load a model
model = YOLO("runs/classify/train/weights/best.pt")  # load a custom model

# Predict with the model
results = model("微信图片_20241113214224.jpg")  # predict on an image