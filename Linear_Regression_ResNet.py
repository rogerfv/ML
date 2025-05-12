#!/usr/bin/env python
# coding: utf-8

# XARXA NEURONAL GRAN

# In[1]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3, 4'


# In[2]:


import numpy as np
import torch
from matplotlib import pyplot as plt
import h5py
import matplotlib.pyplot as plt
import cv2  # OpenCV per redimensionar


# In[3]:


#Triga uns 3 minuts en executar

# Fitxer HDF5 amb les matrius d'imatge. Té 3601 (0.0 fins 360.0) datasets amb matrius de 2048 x 2048, una imatge per cada 0.1 graus.
file_path="intensity_matrix_big.h5"  

if os.path.exists("images.pt") and os.path.exists("labels.pt"):
    #Ja existeixen les imatges i etiquetes. No es tornaran a generar.
    images = torch.load("images.pt")
    labels = torch.load("labels.pt")
else:
    # Llegir totes les matrius i els seus angles
    X = []  # Llista per a les matrius d'imatge
    Y = []  # Llista per als angles

    target_size = (256, 256)  # NOVETAT: mida de sortida per a les imatges

    with h5py.File(file_path, 'r') as f:
        for angle_key in f.keys():
            matrix = f[angle_key][:]
            # Redimensionar la imatge a 256x256
            resized = cv2.resize(matrix, target_size, interpolation=cv2.INTER_AREA)
            X.append(resized)
            Y.append(float(angle_key))  # El nom del dataset és l'angle

    # Convertir llistes a arrays de numpy
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # convert to torch
    images = torch.from_numpy(X)
    labels = torch.from_numpy(Y)

    torch.save(images, "images.pt")
    torch.save(labels, "labels.pt")

print(f'{images.shape=}, {labels.shape=}')



# 
# Només hi ha una imatge per cada 0.1 graus, una CNN de classificació no és la millor opció. Podem fer una regressió per a preveure l'angle d'una imatge, i utilitzar "data augmentation", és a dir, simular inputs artificials per tenir més mostres.

# In[6]:


import torchvision.transforms as T

augmentation = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(256, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.RandomApply([T.GaussianBlur(3)], p=0.3),
    T.RandomAffine(degrees=1, translate=(0.02, 0.02)),
    T.ToTensor()
])


# In[7]:


#Triga uns 5 minuts en executar
if os.path.exists("X_aug.pt") and os.path.exists("Y_aug.pt"):
    #Ja existeixen les imatges i etiquetes augmentades. No es tornaran a generar.
    X_aug = torch.load("X_aug.pt")
    Y_aug = torch.load("Y_aug.pt")

else:
    augmented_images = []
    augmented_labels = []

    images = images.numpy()
    labels = labels.numpy()

    n_augmentations = 30

    for i in range(len(images)):
        img = images[i]
        label = labels[i]

        for _ in range(n_augmentations):
            img_aug = augmentation(img.astype(np.uint8))  #Convertim a uint8 per a la transformació
            augmented_images.append(img_aug)
            augmented_labels.append(label)

    # Tensors finals
    X_aug = torch.stack(augmented_images) 
    Y_aug = torch.tensor(augmented_labels, dtype=torch.float32)


    torch.save(X_aug, "X_aug.pt")
    torch.save(Y_aug, "Y_aug.pt")


# In[8]:


# Per assegurar que l'entrenament té mostres de cada angle, creem aquesta funció
from collections import defaultdict

def split_by_angle(Y_aug, train_frac=0.8, seed=42):
    np.random.seed(seed)
    angle_to_indices = defaultdict(list)

    # Agrupar índexs per angle
    for idx, angle in enumerate(Y_aug):
        angle_to_indices[float(angle)].append(idx)

    # Dividir cada grup
    train_idx, test_idx = [], []

    for indices in angle_to_indices.values():
        indices = np.array(indices)
        np.random.shuffle(indices)
        split = int(len(indices) * train_frac)
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])

    return train_idx, test_idx


# In[9]:


from torch.utils.data import Subset, TensorDataset, DataLoader, random_split

train_idx, test_idx = split_by_angle(Y_aug.numpy(), train_frac=0.8)

import torch.nn.functional as F

# Resize a 224x224 (requisito de ResNet)
X_aug_resized = F.interpolate(X_aug, size=(224, 224), mode='bilinear')

# Repetir el canal 3 veces para simular RGB
X_aug_rgb = X_aug_resized.repeat(1, 3, 1, 1)  # [N, 3, 224, 224]


from torch.utils.data import Subset, TensorDataset, DataLoader, random_split


# Dataset y loaders
full_dataset = TensorDataset(X_aug_rgb, Y_aug)
train_ds = Subset(full_dataset, train_idx)
test_ds = Subset(full_dataset, test_idx)

train_loader_2 = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader_2 = DataLoader(test_ds, batch_size=32)


# In[ ]:


from torchvision import models
import torch.nn as nn

class ResNetForRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.resnet18(pretrained=True)

        # Congelar capas si quieres (opcional)
        for param in self.base.parameters():
            param.requires_grad = False

        # Reemplazar la capa final (fc) para regresión
        self.base.fc = nn.Sequential(
            nn.Linear(self.base.fc.in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # regresión: salida escalar
        )

    def forward(self, x):
        return self.base(x).squeeze(1)  # salida shape [batch]


# In[ ]:
import wandb

# Initialize the run
wandb.init(project="cnn-multigpu-resnet", name="run_01", config={
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
})



import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_2 = ResNetForRegression().to(device)
model_2 = nn.DataParallel(model_2)
optimizer = torch.optim.Adam(model_2.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 100

start = time.time()
for epoch in range(epochs):
    model_2.train()
    running_loss = 0
    for images, targets in train_loader_2:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        preds = model_2(images)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        wandb.log({"train_loss": running_loss/len(train_loader_2), "epoch": epoch+1})
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader_2):.4f}")
torch.save(model_2.state_dict(), "angle_cnn_weights_ResNet.pth")
end = time.time()
print(f"Temps d'entrenament: {end - start:.2f} segons")


# In[ ]:

model_2.eval()
all_preds, all_targets = [], []

with torch.no_grad():
    for images, targets in test_loader_2:
        images, targets = images.to(device), targets.to(device)
        preds = model_2(images)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)


errors = np.abs(all_preds - all_targets)

mae = np.mean(errors)
std = np.std(errors)

print(f"MAE: {mae:.2f}°, Desviació estàndar: {std:.2f}°")

lower = np.percentile(errors, 2.5)
upper = np.percentile(errors, 97.5)

print(f"MAE: {mae:.2f}°, IC 95%: [{lower:.2f}°, {upper:.2f}°]")

print(f"Mediana: {np.median(errors):.2f}°")
print(f"Percentils (25%-75%): {np.percentile(errors, 25):.2f}° – {np.percentile(errors, 75):.2f}°")
print(f"Màxim error: {np.max(errors):.2f}°")


# In[ ]:


import matplotlib.pyplot as plt

model_2.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for images, targets in test_loader_2:
        images, targets = images.to(device), targets.to(device)
        preds = model_2(images)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

plt.figure(figsize=(7, 7))
plt.scatter(all_targets, all_preds, alpha=0.3, s=10)
plt.plot([0, 360], [0, 360], 'r--', label='Ideal')
plt.xlabel("Ángulo real (°)")
plt.ylabel("Ángulo predicho (°)")
plt.title("Predicción vs Valor real")
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:


errors = np.abs(all_preds - all_targets)

plt.figure(figsize=(8, 4))
plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Error absoluto (°)")
plt.ylabel("Frecuencia")
plt.title("Distribución del error absoluto")
plt.grid(True)
plt.show()


# In[ ]:


plt.figure(figsize=(8, 4))
plt.scatter(all_targets, errors, alpha=0.3, s=10)
plt.xlabel("Ángulo real (°)")
plt.ylabel("Error absoluto (°)")
plt.title("Error por ángulo")
plt.grid(True)
plt.show()


# In[ ]:


import pandas as pd
import seaborn as sns

df = pd.DataFrame({'angle': all_targets, 'error': errors})
df['angle_group'] = (df['angle'] // 10) * 10  # grupos de 10°

plt.figure(figsize=(12, 5))
sns.boxplot(x='angle_group', y='error', data=df)
plt.xlabel("Grupo de ángulo (°)")
plt.ylabel("Error absoluto (°)")
plt.title("Boxplot del error por grupo de ángulo")
plt.grid(True)
plt.show()

