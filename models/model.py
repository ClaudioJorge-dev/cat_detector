import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

model = None
device = None

def load_model():
    global model, device

    if model is None:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

    return model