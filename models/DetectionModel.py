from ultralytics import YOLO
from pathlib import Path
import cv2

class DetectionModel:
    def __init__(self, model_path="models/yolo/yolov8n.pt"):
        # use nano model (faster)
        self.model = YOLO(model_path)
        
    def detect_folder(self, folder_path):
        results_list = []
        
        for img_path in Path(folder_path).glob("*.[jp][pn]g"): 
            images, detections = self.detect(str(img_path))
            results_list.append((images, detections))
            
        return results_list

    def detect(self, image_path):
        results = self.model(image_path)
        image = cv2.imread(image_path)
        
        detections = []
        
        for result in results:
            for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                
                if conf > 0.5 and int(cls) == 15:  # Only draw boxes with confidence greater than 0.5 and class equal to 15 (cat)
                    
                    x1, y1, x2, y2 = map(int, box)
                    # crops the cat to be used by the mobilenet model
                    cat_crop = image[y1:y2, x1:x2]
                    
                    detections.append(
                        {
                            "box": (x1, y1, x2, y2),
                            "confidence": float(conf),
                            "croped_image": cat_crop
                        }
                    )
                    
        return image, detections
                