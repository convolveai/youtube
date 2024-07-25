from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov10n.pt")

results = model.track(source="./output.mp4", show=True,  tracker="bytetrack.yaml", classes=[0], save=True)
