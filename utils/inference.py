import os, json
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from utils.gradcam import GradCAM, overlay_heatmap_on_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_label_map(path="models/label_map.json"):
    if os.path.exists(path):
        with open(path, "r") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    # fallback
    return {0: "NORMAL", 1: "PNEUMONIA"}

def load_model(weights_path="models/best_model.pt", model_name="densenet121"):
    label_map = load_label_map()
    num_classes = len(label_map)
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes, in_chans=3)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval().to(DEVICE)
    return model, label_map

def get_target_layer(model):
    """
    Try to find the last convolutional layer for Grad-CAM.
    Works for many timm CNNs; for DenseNet, we use model.features.denseblock4
    """
    try:
        return model.features[-1]  # may not exist on some
    except Exception:
        pass
    # DenseNet-like
    if hasattr(model, "features") and hasattr(model.features, "denseblock4"):
        return model.features.denseblock4
    # Fallback: search for last Conv2d
    last_conv = None
    for name, module in model.named_modules():
        if module.__class__.__name__ == "Conv2d":
            last_conv = module
    if last_conv is None:
        raise RuntimeError("Could not find a conv layer for Grad-CAM.")
    return last_conv

def preprocess(image: Image.Image, img_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return tfm(image).unsqueeze(0)

def predict_with_cam(img_path: str, img_size=224, model_name="densenet121"):
    model, label_map = load_model(model_name=model_name)
    img = Image.open(img_path).convert("RGB")
    input_tensor = preprocess(img, img_size=img_size).to(DEVICE)

    # forward
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy().squeeze(0)

    # Grad-CAM on predicted class
    target_layer = get_target_layer(model)
    cam = GradCAM(model, target_layer)
    heatmap = cam(input_tensor, class_idx=int(probs.argmax()))

    # prepare overlay
    img_resized = img.resize((img_size, img_size))
    overlay = overlay_heatmap_on_image(np.array(img_resized), heatmap)

    # format outputs
    idx2label = {i: name for i, name in label_map.items()}
    probs_dict = {idx2label[i]: float(p) for i, p in enumerate(probs)}
    top_label = idx2label[int(probs.argmax())]
    return top_label, probs_dict, overlay
