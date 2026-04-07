from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from imagenet_load import get_label
from PredictionObj import PredictionObj

# resize and normalize the image
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

model = None

def load_and_transform(image_path):
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def load_model():
    global model
    global device
    if model is None:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()
    return model

def predict_batch(image_paths):
    images = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        images = list(tqdm(executor.map(load_and_transform, image_paths), total=len(image_paths)))
    
    images = [img.to(device) for img in images]
    batch = torch.stack(images)
    
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
    return probs.cpu()


