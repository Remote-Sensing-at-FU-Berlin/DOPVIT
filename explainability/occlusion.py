import torch
import numpy as np


def band_occlusion(model, image, target_class, patch_size=8):
    """
    Occlusion sensitivity for multispectral images.
    """
    model.eval()

    _, H, W = image.shape
    heatmap = np.zeros((H, W))

    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):

            img_occluded = image.clone()
            img_occluded[:, y:y+patch_size, x:x+patch_size] = 0

            logits, _ = model(img_occluded.unsqueeze(0))
            score = logits[0, target_class].item()

            heatmap[y:y+patch_size, x:x+patch_size] = score

    return heatmap
