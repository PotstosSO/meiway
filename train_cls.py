import torch
# from pandas.tests.tools.test_to_datetime import epochs

from ultralytics import YOLO

# Load a model
print(torch.cuda.is_available())
model = YOLO("yolov8n.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="./data/silkworm", epochs=100, imgsz=64, batch=16)


