import cv2
from pathlib import Path
from models.ClassificationModel import ClassificationModel
from models.DetectionModel import DetectionModel
import os

# Welcoming message
print("Welcome to the Cat Breed Detection and Classification System!")

# Models initialization
detector = DetectionModel()
classifier = ClassificationModel()

# Path can be a single image or a directory containing multiple images
PATH = input("Enter the path to the image or directory: ") or "images/default.jpg"
is_folder = Path(PATH).is_dir()

generate_website = input("Do you want to generate a website to display results? (y/n): ").lower() == "y"
OUTPUT_PATH = input("Enter the output path for results (default: 'outputs'): ") or "outputs"

if not Path(OUTPUT_PATH).exists():
    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

# Always get a list of (image, detections)
results = detector.detect_folder(PATH) if is_folder else [detector.detect(PATH)] 

for idx, (image, detections) in enumerate(results):
    for det in detections:
        result = classifier.predict(det["croped_image"])
        
        if result:
            x1,y1,x2,y2 = det["box"]
        
            # draw the bounding box on the image
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(image, result.label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,250,0), 2)
            cv2.putText(image, f"{result.probability*100:.2f}%", (x1, y2+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,250,0), 2)
    # Save output for each image separately
    output_name = f"{OUTPUT_PATH}/{Path(PATH).name}" if not is_folder else f"{OUTPUT_PATH}/output_{idx}.jpg"
    cv2.imwrite(output_name, image)
    idx += 1