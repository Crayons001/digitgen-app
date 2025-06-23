import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    def __init__(self, input_dim=784, label_dim=10, latent_dim=20):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(input_dim + label_dim, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim + label_dim, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def encode(self, x, y):
        h = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h = F.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, y), mu, logvar
