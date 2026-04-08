import cv2
import torch
from torchvision import transforms
from utils import get_cat_breed_from_probs
from models.model import load_model
from pathlib import Path

class ClassificationModel:
    def __init__(self):
        self.model = load_model()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        
    def predict(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image).unsqueeze(0)
        
        device = next(self.model.parameters()).device
        image = image.to(device)
        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return get_cat_breed_from_probs(probs[0])