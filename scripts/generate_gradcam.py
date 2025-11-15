import torch
import matplotlib.pyplot as plt
from models.builder import get_model
from explainability.gradcam import gradcam_cnn
from data.datasets import TIFDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_model("cnn", num_classes=2, input_channels=4)
model.load_state_dict(torch.load("model.pth"))
model.to(device)

dataset = TIFDataset(...)
image, label = dataset[0]
image = image.unsqueeze(0).to(device)

cam = gradcam_cnn(model, image, target_class=label)

plt.imshow(cam.cpu().numpy(), cmap="viridis")
plt.colorbar()
plt.savefig("gradcam.png")
