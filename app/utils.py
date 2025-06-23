import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from app.cvae_model import CVAE

MODEL_PATH = "cvae_mnist.pth"
IMAGE_DIR = "static/generated"

device = torch.device("cpu")
model = CVAE().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

def generate_images(digit, num_samples=5):
    z = torch.randn(num_samples, 20)
    y = F.one_hot(torch.tensor([digit]*num_samples), num_classes=10).float()
    generated = model.decode(z, y).view(-1, 28, 28).detach().numpy()

    filenames = []
    for i, img in enumerate(generated):
        path = os.path.join(IMAGE_DIR, f'digit_{digit}_{i}.png')
        plt.imsave(path, img, cmap='gray')
        filenames.append(path)

    return filenames
