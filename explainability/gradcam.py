import torch
import torch.nn.functional as F

def gradcam_cnn(model, image, target_class):
    """
    Grad-CAM for CNN-based models.
    """
    model.eval()

    activations = []
    gradients = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Attach hooks to final conv layer
    last_conv = model.features[-1]
    h1 = last_conv.register_forward_hook(forward_hook)
    h2 = last_conv.register_backward_hook(backward_hook)

    logits, _ = model(image)
    score = logits[:, target_class]
    score.backward()

    act = activations[0]
    grad = gradients[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1)
    cam = F.relu(cam)

    h1.remove()
    h2.remove()

    return cam
