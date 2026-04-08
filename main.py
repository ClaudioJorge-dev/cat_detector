import cv2
from pathlib import Path
from ClassificationModel import ClassificationModel
from detection.DetectionModel import DetectionModel
import os

# Path can be a single image or a directory containing multiple images
PATH = "images"  

detector = DetectionModel()
classifier = ClassificationModel()

is_folder = Path(PATH).is_dir()

# Always get a list of (image, detections)
results = detector.detect_folder(PATH) if is_folder else [detector.detect(PATH)] 
idx = 0

for image, detections in results:
    for det in detections:
        result = classifier.predict(det["croped_image"])
        
        if result:
            x1,y1,x2,y2 = det["box"]
        
            # draw the bounding box on the image
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(image, result.label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,250,0), 2)
            
    # Save output for each image separately
    output_name = f"output_{Path(PATH).name}" if not is_folder else f"output_{idx}.jpg"
    cv2.imwrite(output_name, image)
    idx += 1