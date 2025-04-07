from ultralytics import YOLO


model = YOLO('yolov8-seg.yaml').load('yolov8n-seg.pt')  #改成自己所放的位置
model.train(data='./data/silkworm-seg.yaml',epochs=50,imgsz=640,batch=4)
