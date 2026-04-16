from sys import path

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


"""
Used to create model for training and loading the model for inference.
"""
def create_model():
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(1280,12) # Adjust the final layer to output 12 classes (for 12 cat breeds)
    return model


def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    return model

def get_inference_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_model(model, path, device)
    model.eval()
    return model, device