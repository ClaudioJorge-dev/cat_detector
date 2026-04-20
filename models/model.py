from sys import path

import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


"""
Used to create model for training and loading the model for inference.
"""
def create_model(training = False):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(1280,12) # Adjust the final layer to output 12 classes (for 12 cat breeds)
    
    if training:
        # Freeze everything first
        for param in model.features.parameters():
            param.requires_grad = False

        # Unfreeze last 20 blocks
        for param in model.features[-20:].parameters():
            param.requires_grad = True
    
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