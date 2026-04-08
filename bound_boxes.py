from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

# results and image should be the same image
results = model("images/2cats1dog.jpg")
image = cv2.imread("images/2cats1dog.jpg")

for result in results:
    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        if conf > 0.5 and cls == 15:  # Only draw boxes with confidence greater than 0.5 and class equal to 15 (cat)
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2) # green, 2px
        
cv2.imwrite("output.jpg", image)