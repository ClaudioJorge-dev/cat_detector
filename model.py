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

def load_model():
    global model
    if model is None:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()
    return model

# predict the class of the image
def predict(image_path):
    # load image
    image = Image.open(image_path)
    img_transformed = transform(image).unsqueeze(0)
    
    # predict the class
    with torch.no_grad():
        outputs = model(img_transformed)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        
    top5_prob, top5_catid = torch.topk(probs, 5)
    results = []
    for i in range(top5_prob.size(0)):
        obj = PredictionObj(
            cat_class=top5_catid[i].item(),
            probability=top5_prob[i].item(),
            label=get_label(top5_catid[i].item())
        )
        results.append(obj)
        
    return results


