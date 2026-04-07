from utils import get_cat_breed_from_probs
from model import load_model, predict_batch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor 
import os

class CatDetector:
    
    def __init__(self, path):
        self.path = path or "No image path provided."
        self.model = load_model()
        
    def detect_cat(self, path=None):
        path = path or self.path
        
        if not self.model:
            print("Failed to load the model.")
            return
        
        if path == "No image path provided." or Path(path).exists() == False:
            print(path)
            return
        
        if Path(path).is_dir():
            self.detect_all_batch()
        else:
            self.detect_all_batch([self.path])
            
    def detect_all_batch(self, paths=None):
        paths = paths or [
            p for p in Path(self.path).glob("*")
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]

        probs = predict_batch(paths)

        for i, path in enumerate(paths):
            results = get_cat_breed_from_probs(probs[i])
            self.process_results(results, path)
            
    
    def process_results(self, results, image_path=None):
        image_name = os.path.basename(image_path) if image_path else "Unknown Image"
        if results:
            print(f"For \"{image_name}\": {results.label} with probability {results.probability*100:.2f}%")
        else:
            print(f"For \"{image_name}\": No cat detected.")