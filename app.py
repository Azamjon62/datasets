from ultralytics import YOLO
from ultralytics import settings

# print(settings)

# settings.update({"datasets_dir": "/Users/vawzen/Documents/AI projects/cooler-shelf-detector/dataset"})
# settings.update({"weights_dir": "/Users/vawzen/Documents/AI projects/cooler-shelf-detector"})
# settings.update({"runs_dir": "/Users/vawzen/Documents/AI projects/cooler-shelf-detector"})


model = YOLO("yolov8n-seg.pt")

model.train(data="dataset/data.yaml", imgsz=960, batch=2, epochs=50, patience=5, device='mps')

