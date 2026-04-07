from CatDetector import CatDetector

# Path can be a single image or a directory containing multiple images
PATH = "images/siamese_cat.jpg"  

detector = CatDetector(PATH)
detector.detect_cat()