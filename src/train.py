import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import glob
from copy import deepcopy
#Local class
from model import IncisionSeedModel
from dataset import IncisionDataset

# --- GLOBAL CONFIGURATION ---
IM_RESIZE = 240
BATCH_SIZE = 16
MIN_VAL_ERROR = 0.01  # Early Stopping Threshold
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path must be adjusted if required
IMAGE_PATH = "data/images/*.jpg"
LABEL_PATH = "data/labels/*.txt"

def train_model(model, train_loader, val_loader, train_dataset, val_dataset,
                optimizer, num_epochs, min_val_loss_threshold=float('inf')):
    """
    Train loop with validation and Early Stopping.
    """
    criterion = nn.MSELoss() 
    best_model_wts = deepcopy(model.state_dict())
    best_val_loss = float('inf') 

    for epoch in range(num_epochs):
        # --- TRAIN PHASE ---
        model.train() 
        loss_epoch_train = 0.0
        
        for img, labels in train_loader:
            img, labels = img.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loss_epoch_train += loss.item() * labels.size(0)

        loss_epoch_train /= len(train_dataset)

        # --- VALIDATION PHASE ---
        model.eval()
        loss_epoch_val = 0.0
        val_error_sum = 0.0
        
        with torch.no_grad():
            for img, labels in val_loader:
                img, labels = img.to(DEVICE), labels.to(DEVICE)
                outputs = model(img)
                loss = criterion(outputs, labels)
                loss_epoch_val += loss.item() * labels.size(0)
                # Euclidean distance (L2)
                val_error_sum += F.pairwise_distance(outputs, labels, p=2).sum().item()

        loss_epoch_val /= len(val_dataset)
        val_distance_error = val_error_sum / len(val_dataset)
        
        # Save the best model
        if loss_epoch_val < best_val_loss:
            best_val_loss = loss_epoch_val
            best_model_wts = deepcopy(model.state_dict())
            os.makedirs("models", exist_ok=True)
            torch.save(best_model_wts, "models/best_model.pth")

        print(f"Epoch {epoch+1}/{num_epochs} | Train MSE: {loss_epoch_train:.5f} | Val MSE: {loss_epoch_val:.5f} | Dist Err: {val_distance_error:.5f}")

        # Early Stopping
        if loss_epoch_val <= min_val_loss_threshold:
            print(f"--> Early Stopping: Threshold {min_val_loss_threshold} achieved.")
            break

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # Prepare paths and Split
    all_images = sorted(glob.glob(IMAGE_PATH))
    all_labels = sorted(glob.glob(LABEL_PATH))

    train_img, temp_img, train_lab, temp_lab = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
    val_img, test_img, val_lab, test_lab = train_test_split(temp_img, temp_lab, test_size=0.5, random_state=42)

    # 2. DataLoaders
    train_ds = IncisionDataset(train_img, train_lab, resize_size=IM_RESIZE)
    val_ds = IncisionDataset(val_img, val_lab, resize_size=IM_RESIZE)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Initialize model
    model = IncisionSeedModel(freeze_backbone=True).to(DEVICE)

    # --- PHASE 1: Backbone Freezed ---
    print("\nIniciando Fase 1: Entrenamiento de capa final...")
    optimizer_f1 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    model = train_model(model, train_loader, val_loader, train_ds, val_ds, optimizer_f1, num_epochs=300, min_val_loss_threshold=MIN_VAL_ERROR)

    # --- PHASE 2: Full Net Fine-Tuning ---
    print("\nIniciando Fase 2: Fine-Tuning completo...")
    model.unfreeze()
    optimizer_f2 = Adam(model.parameters(), lr=1e-5) # LR mucho más bajo
    model = train_model(model, train_loader, val_loader, train_ds, val_ds, optimizer_f2, num_epochs=1000, min_val_loss_threshold=MIN_VAL_ERROR)

    print("\nTraining done succesful. Model saved in models/best_model.pth")
