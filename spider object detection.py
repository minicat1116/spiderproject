from ultralytics import YOLO


#load a model
model = YOLO("yolov8n.yaml")

#use the model
results = model.train (data="config.yaml" , epochs=50)
