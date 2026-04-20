from torchvision.datasets import OxfordIIITPet # Pet dataset for training
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from model import create_model, load_model
import torch

def validate(model, val_loader, criterion, device):
    """
    Validates the model in training with the validation set.
    REturns the average loss over the validation set.
    """
    model.eval() # set the model to evaluation mode
    total_loss = 0.0
    
    with torch.no_grad(): # no need to compute gradients during validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device) # move to the same device as the model
            outputs = model(images) # forward pass (Calculates the predicted probabilities and the resulting loss (how wrong the model is).)
            loss = criterion(outputs, labels) # compute the loss
            total_loss += loss.item() # accumulate the loss
            
    model.train()
    return total_loss / len(val_loader) # return the average loss over the validation set

"""
Training loop
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
initial_learning_rate = 5e-5
batch_size = 32 
num_epochs = 50
scheduler_patience = 3 # number of epoch to wait before saving the best model
best_val_loss = float('inf') 
patience_counter = 0 # tracks how long the validation loss has not improved (used for early stopping)
early_stop_patience = 5 # number of epoch to wait before stopping the training if the validation loss does not improve

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    
    # Data augmentation
    transforms.RandomHorizontalFlip(), # randomly flip the image horizontally (default = 50% chance)
    transforms.RandomRotation(degrees=10), # randomly rotate the image by up to 10 degrees
    transforms.ColorJitter(
        brightness = 0.2, contrast = 0.2, saturation = 0.25,
        hue = 0.05
    ), # randomly change the brightness, contrast, saturation, and hue of the image (with specified ranges) > https://docs.pytorch.org/vision/main/generated/torchvision.transforms.ColorJitter.html
    
    # Tranform to tensor before normalization
    transforms.ToTensor(),
    
    # Normalization (scales pixel values to a consistent range, most commonly [0,1] or [-1,1]. This improves model convergence, helps gradients behave, and often results in better accuracy.)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization values
])

# Load the dataset. Followed Doc: https://docs.pytorch.org/vision/main/generated/torchvision.datasets.OxfordIIITPet.html

dataset = OxfordIIITPet(root='data', # path to store the dataset 
                        split='trainval', # use the training and validation split
                        download=False, # download the dataset (False if already downloaded)
                        transform=transforms 
                        )

# filter only cats
cat_data = [(img, labels) for img, labels in dataset if labels < 12] # Dog breeds start at index 12, so we take only those with labels less than 12

"""
Split the dataset into training and validation sets
Training data: Used to train the model (80% of the data)
Validation data: Used to evaluate the model during training and tune hyperparameters (20% of the data)

This is ensures that the model generalizes well to unseen data and to prevent overfitting (where the model performs well on training data but poorly on new, unseen data).
"""

train_data_size = int(0.8 * len(cat_data)) # 80% for training
val_data_size = len(cat_data) - train_data_size # remaining 20% for validation

# shuffles the dataset and splits it into training and validation sets based on the specified sizes.
# this allows the model to learn from a diverse set of examples and not only the first 80% breeds
# random_split dont ensure that the same breeds are in both training and validation sets, but it does ensure that the data is randomly distributed between the two sets, which helps in generalization.
train_data, val_data = random_split(cat_data, [train_data_size, val_data_size]) 

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # shuffle the training data for better learning
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) # no need to shuffle validation

model = create_model(training=True) # create the model with training=True to freeze early layers
model = model.to(device)
model.train() # set the model to training mode

# Computes the cross entropy loss 
# (quantifying the difference between predicted probability distributions and actual true labels) 
# between input and target 
# https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
criterion = torch.nn.CrossEntropyLoss() 

# https://docs.pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
optimizer = torch.optim.Adam(model.parameters(), lr=initial_learning_rate) # The learning rate controls how big each “step” is when the model learns from its mistakes.

# ReduceLROnPlateau reduces the learning rate when a metric has gotten worse for patience epochs.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=scheduler_patience, factor=0.1) 

# epoch definition: https://www.geeksforgeeks.org/machine-learning/epoch-in-machine-learning/
for epoch in range(num_epochs): # loop over the dataset mult times
    
    total_train_loss = 0.0
    
    for images, labels in train_loader:
        optimizer.zero_grad() # zero the parameter gradients
        #Ensures everything is on the same device
        images, labels = images.to(device), labels.to(device) # move to the same device as the model
        outputs = model(images) # forward pass (Calculates the predicted probabilities and the resulting loss (how wrong the model is).)
        loss = criterion(outputs, labels) # compute the loss
        loss.backward() # backward pass/backpropagation (Computes the gradient of the loss with respect to the model parameters.)
        optimizer.step() # update the model parameters based on the computed gradients (using the Adam optimization algorithm).

        total_train_loss += loss.item() # accumulate the training loss
    
    train_loss = total_train_loss / len(train_loader) # average training loss over the epoch
    val_loss = validate(model, val_loader, criterion, device) # compute validation loss
    scheduler.step(val_loss) # update the learning rate based on the validation loss (if it has stopped improving)
    
    print(f"Epoch {epoch +1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']}") 
    
    # save the best model
    if val_loss<best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'trained_models/cat_pred_model.pth')
        print(f"Best model saved at epoch: {epoch + 1} with validation loss: {best_val_loss:.4f}")
        patience_counter = 0 
    else:
        patience_counter += 1
        
    # early stopping
    if patience_counter >= early_stop_patience:
        print(f"Early stopping triggered at epoch: {epoch + 1}. No improvement in validation loss for {early_stop_patience} epochs.")
        break
    