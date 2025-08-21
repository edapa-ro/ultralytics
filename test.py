import ultralytics

model_path = "MODELS/adown_backbone/yolov5nC3k2GhostC2f.yaml"

dataset = "DATASET/detect.yaml"
model = ultralytics.YOLO(model_path).load("MODELS/yolov5n1024_coco12.pt")

model.train(data=dataset, epochs=100, imgsz=1024, batch=16, device="0", workers=8)