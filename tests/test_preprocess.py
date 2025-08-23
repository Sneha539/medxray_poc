import os
from PIL import Image
import numpy as np
from utils.inference import preprocess

def test_preprocess_shape(tmp_path):
    # Create a fake RGB image
    img = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
    out = preprocess(img, img_size=224)
    assert out.shape == (1, 3, 224, 224)
