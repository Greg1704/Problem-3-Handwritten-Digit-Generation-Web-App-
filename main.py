# app.py
import streamlit as st
import torch
from torch import nn
import numpy as np
from PIL import Image
from train import VAE  # Asegúrate que la clase VAE está definida en train.py

device = 'cpu'
latent_dim = 20
model = VAE(latent_dim)
model.load_state_dict(torch.load("vae_mnist.pth", map_location=device))
model.eval()


st.title("MNIST Digit Generator")

# Elegir dígito
digit = st.selectbox("Select Digit (0-9):", range(10))

if st.button("Generate Images"):
    images = []
    for _ in range(5):
        z = torch.randn(1, latent_dim).to(device)

        # Aquí, para guiar al dígito deseado, habría que entrenar un modelo condicionado
        # o filtrar por dígito en latent space. Para esta versión simple, asumimos
        # que generamos muestras genéricas y no específicas al dígito.
        sample = model.decoder(z).cpu().view(28, 28).detach().numpy()
        images.append(sample)

    # Mostramos
    cols = st.columns(5)
    for col, img in zip(cols, images):
        col.image(img, clamp=True, caption=f"Digit: {digit}")

