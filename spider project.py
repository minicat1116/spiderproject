from ultralytics import YOLO
import cv2

#load YOLOv8 model
model = YOLO('yolov8n.pt')

#video path
video_path = '/Users/janhavi/Desktop/tigra_hunting.mov'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #object detection & tracking
    results = model.track(frame, persist=True)

    #plot results
    frame_ = results[0].plot()

    #display
    cv2.imshow('YOLOv8 Tracking', frame_)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
