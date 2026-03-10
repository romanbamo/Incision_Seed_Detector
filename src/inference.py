import cv2
import torch
import numpy as np
from model import IncisionSeedModel
from torchvision import transforms

def predict_seed(image_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    model = IncisionSeedModel(freeze_backbone=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()

    # 2. Preprocess Image
    img_orig = cv2.imread(image_path)
    h, w, _ = img_orig.shape
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (240, 240))
    
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img).unsqueeze(0).to(device)

    # 3. Inference
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]
    
    # 4. De-normalize coordinates
    x_pred, y_pred = int(output[0] * w), int(output[1] * h)
    
    return (x_pred, y_pred), img_orig

if __name__ == "__main__":
    # Use
    coords, img = predict_seed("data/sample_images/test.jpg", "models/best_model.pth")
    print(f"Predicted Seed at: {coords}")
    
    # Draw and show
    cv2.circle(img, coords, 10, (0, 0, 255), -1)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
