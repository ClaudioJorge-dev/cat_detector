import time
from CatDetector import CatDetector

# Path can be a single image or a directory containing multiple images
PATH = "images"  

init = time.time()
detector = CatDetector(PATH)
detector.detect_cat()
print("Time taken: ", time.time() - init)