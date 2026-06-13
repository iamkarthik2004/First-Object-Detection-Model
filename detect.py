from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Run object detection
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True
)

print("Detection completed!")