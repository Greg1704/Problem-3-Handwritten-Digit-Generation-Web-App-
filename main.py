# app.py
import streamlit as st
import torch
from torch import nn
import numpy as np
from PIL import Image

# --- Definición de la clase VAE (igual que en training) ---
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU()
        )
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


# --- Setup ---
device = 'cpu'
latent_dim = 20

@st.cache_resource  # para que no cargue el modelo cada vez
def load_model():
    model = VAE(latent_dim)
    model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
    model.eval()
    return model

model = load_model()

st.title("Generador de Dígitos Manuscritos (MNIST)")

# Selector de dígito (0-9)
digit = st.selectbox("Selecciona un dígito (0-9):", list(range(10)))

# Botón para generar imágenes
if st.button("Generar 5 imágenes"):
    st.write(f"Generando 5 imágenes para el dígito: {digit} (modelo no condicional)")

    cols = st.columns(5)

    for i in range(5):
        # Generar vector latente aleatorio
        z = torch.randn(1, latent_dim).to(device)

        # Generar imagen desde el decoder
        with torch.no_grad():
            img = model.decoder(z).cpu().view(28, 28).numpy()

        # Normalizar y convertir a imagen para mostrar
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8, mode="L")

        cols[i].image(pil_img, caption=f"Imagen {i+1}", use_column_width=True)
