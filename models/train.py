from torchvision.datasets import OxfordIIITPet # Pet dataset for training
from torchvision import transforms
from torch.utils.data import DataLoader
from model import create_model, load_model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# Load the dataset. Followed Doc: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html

dataset = OxfordIIITPet(root='data', # path to store the dataset 
                        split='trainval', # use the training and validation split
                        download=False, # download the dataset (False if already downloaded)
                        transform=transforms 
                        )


# filter only cats
cat_data = [(img, labels) for img, labels in dataset if labels < 12] # Dog breeds start at index 12, so we take only those with labels less than 12

loader = DataLoader(cat_data, batch_size=32, shuffle=True)

model = create_model()
model = model.to(device)
model.train() # set the model to training mode

# Computes the cross entropy loss 
# (quantifying the difference between predicted probability distributions and actual true labels) 
# between input and target 
# https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
criterion = torch.nn.CrossEntropyLoss() 

# https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # The learning rate controls how big each “step” is when the model learns from its mistakes.

################################# TODO scheduler to adjust learning rate during training


# epoch definition: https://www.geeksforgeeks.org/machine-learning/epoch-in-machine-learning/
for epoch in range(20): # loop over the dataset mult times
    for images, labels in loader:
        optimizer.zero_grad() # zero the parameter gradients
        #Ensures everything is on the same device
        images, labels = images.to(device), labels.to(device) # move to the same device as the model
        outputs = model(images) # forward pass (Calculates the predicted probabilities and the resulting loss (how wrong the model is).)
        loss = criterion(outputs, labels) # compute the loss
        loss.backward() # backward pass/backpropagation (Computes the gradient of the loss with respect to the model parameters.)
        optimizer.step() # update the model parameters based on the computed gradients (using the Adam optimization algorithm).
        
    print(f"Epoch {epoch +1}/10, Loss: {loss.item():.4f}") # print the loss for the current epoch
    
# save model
torch.save(model.state_dict(), 'cat_classifier.pth') # pth = PyTorch model file extension