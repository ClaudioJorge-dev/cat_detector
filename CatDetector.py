from utils import get_cat_breed
from model import predict, load_model
from pathlib import Path
import os

class CatDetector:
    _loaded_model = None
    
    def __init__(self, path):
        self.path = path or "No image path provided."
        
    def _load_model(self):
        if not CatDetector._loaded_model:
            CatDetector._loaded_model = load_model()
        return CatDetector._loaded_model
        
    def detect_cat(self, path=None):
        path = path or self.path
        
        if not self._load_model():
            print("Failed to load the model.")
            return
        
        if path == "No image path provided." or Path(path).exists() == False:
            print(path)
            return
        
        if Path(path).is_dir():
            for image_path in Path(path).glob("*.*"):
                self.detect_cat(image_path)
        else:
            print(f"Processing: {path}...")
            preds = predict(path)
            results = get_cat_breed(preds)
            self.process_results(results, path)
    
    def process_results(self, results, image_path=None):
        image_name = os.path.basename(image_path) if image_path else "Unknown Image"
        if results:
            print(f"For \"{image_name}\": {results.label} with probability {results.probability*100:.2f}% \n")
        else:
            print(f"For \"{image_name}\": No cat detected. \n")