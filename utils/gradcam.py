import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Minimal Grad-CAM for CNNs.
    Usage:
        cam = GradCAM(model, target_layer)
        heatmap = cam(img_tensor, class_idx=None)  # returns HxW np.float32 in [0,1]
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = probs.argmax(dim=1).item()

        loss = logits[:, class_idx].sum()
        loss.backward()

        # activations: [B, C, H, W]; gradients: [B, C, H, W]
        grads = self.gradients  # last conv gradients
        acts = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)  # global-average-pool over H,W

        cam = (weights * acts).sum(dim=1, keepdim=True)  # [B,1,H,W]
        cam = F.relu(cam)
        cam = cam.squeeze(0).squeeze(0)  # [H,W]
        cam -= cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

def overlay_heatmap_on_image(img, heatmap):
    # Ensure both are the same size (resize heatmap to match image)
    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # Blend image + heatmap
    overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)
    return overlay

