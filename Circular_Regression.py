#!/usr/bin/env python
# coding: utf-8

# XARXA NEURONAL GRAN

# In[1]:

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8, 9'


# In[2]:


import numpy as np
import torch
from matplotlib import pyplot as plt
import h5py
import matplotlib.pyplot as plt
import cv2  # OpenCV per redimensionar


# In[3]:


#Triga uns 3 minuts en executar, per això carreguem les imatges i labels

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

# Només hi ha una imatge per cada 0.1 graus, una CNN de classificació no és la millor opció. Podem fer una regressió per a preveure l'angle d'una imatge, i utilitzar "data augmentation", és a dir, simular inputs artificials per tenir més mostres.

# In[4]:


import torchvision.transforms as T

augmentation = T.Compose([
    T.ToPILImage(),
    T.RandomResizedCrop(256, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
    T.ColorJitter(brightness=0.3, contrast=0.3),
    T.RandomApply([T.GaussianBlur(3)], p=0.3),
    T.RandomAffine(degrees=1, translate=(0.02, 0.02)),
    T.ToTensor()
])


# In[5]:


#Triga uns 5 minuts en executar, de nou, carreguem les imatges i labels

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


# In[6]:


def encode_angles_deg(y_deg):
    """Convierteix angles en graus a (cosθ, sinθ)"""
    radians = np.deg2rad(y_deg)
    cos_theta = np.cos(radians)
    sin_theta = np.sin(radians)
    return np.stack([cos_theta, sin_theta], axis=1)  # shape: [N, 2]

Y_aug_encoded = torch.tensor(encode_angles_deg(Y_aug.numpy()), dtype=torch.float32)


# In[9]:


# Per assegurar que l'entrenament té mostres de cada angle, creem aquesta funció
from collections import defaultdict

def split_by_angle(Y_aug_encoded, train_frac=0.8, seed=42):
    np.random.seed(seed)
    angle_to_indices = defaultdict(list)

    # Calcular ángulo en grados de cada vector (cosθ, sinθ)
    Y_aug_encoded = Y_aug_encoded.numpy() if isinstance(Y_aug_encoded, torch.Tensor) else Y_aug_encoded
    angles_deg = (np.rad2deg(np.arctan2(Y_aug_encoded[:, 1], Y_aug_encoded[:, 0])) % 360).round(2)

    # Agrupar índices por ángulo
    for idx, angle in enumerate(angles_deg):
        angle_to_indices[angle].append(idx)

    # Dividir cada grupo
    train_idx, test_idx = [], []
    for indices in angle_to_indices.values():
        indices = np.array(indices)
        np.random.shuffle(indices)
        split = int(len(indices) * train_frac)
        train_idx.extend(indices[:split])
        test_idx.extend(indices[split:])

    return train_idx, test_idx


# In[ ]:


## Funcions per decodificar i evaluar els models
def decode_angles(preds_tensor):
    preds_np = preds_tensor.detach().cpu().numpy()
    angles_rad = np.arctan2(preds_np[:, 1], preds_np[:, 0])
    return np.rad2deg(angles_rad) % 360

def circular_mae(pred_deg, true_deg):
    diff = np.abs(pred_deg - true_deg)
    return np.mean(np.minimum(diff, 360 - diff))


# ## Carregar dades model 1

# In[ ]:


from torch.utils.data import Subset, TensorDataset, DataLoader, random_split

train_idx, test_idx = split_by_angle(Y_aug_encoded.numpy(), train_frac=0.8)

# Dataset y loaders
full_dataset = TensorDataset(X_aug, Y_aug_encoded)
train_ds = Subset(full_dataset, train_idx)
test_ds = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=32, num_workers=4, pin_memory=True)


# ## Arquitectura Model 1

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F

class AngleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 256→128
        x = self.pool(F.relu(self.conv2(x)))  # 128→64
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)  # Output shape: [batch]


# ## Entrenament model 1

# In[ ]:

import wandb

# Initialize the run
wandb.init(project="cnn-multigpu-circular", name="run_01", config={
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 1e-3,
})

import time
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AngleCNN()
model = nn.DataParallel(model)  #Si tenim més d'una GPU, utilitzem DataParallel
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def circular_loss(preds, targets):
    return loss_fn(preds[:, 0], targets[:, 0]) + loss_fn(preds[:, 1], targets[:, 1])





if os.path.exists("angle_cnn_weights_circular.pth"):
    state_dict = torch.load("angle_cnn_weights_circular.pth")
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})


else:
    start = time.time()

    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            preds = model(images)
            loss = circular_loss(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            wandb.log({"train_loss": total_loss/len(train_loader), "epoch": epoch+1})
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "angle_cnn_weights_circular.pth")

    end = time.time()
    print(f"Temps d'entrenament: {end - start:.2f} segons")



# ## Avaluació model 1

# In[ ]:


all_preds = []
all_targets = []

model.eval()
with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)

        preds = model(images)
        all_preds.append(decode_angles(preds))

        target_angles = torch.atan2(targets[:, 1], targets[:, 0]) * 180 / np.pi
        target_angles = target_angles % 360
        all_targets.append(target_angles.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

#mae_circ = circular_mae(all_preds, all_targets)
errors = np.abs(all_preds - all_targets)
errors = np.minimum(errors, 360 - errors)  # corrección circular

mae = np.mean(errors)
std = np.std(errors)

print(f"MAE: {mae:.2f}°, Desviació estàndar: {std:.2f}°")

lower = np.percentile(errors, 2.5)
upper = np.percentile(errors, 97.5)

print(f"MAE: {mae:.2f}°, IC 95%: [{lower:.2f}°, {upper:.2f}°]")

print(f"Mediana: {np.median(errors):.2f}°")
print(f"Percentils (25%-75%): {np.percentile(errors, 25):.2f}° – {np.percentile(errors, 75):.2f}°")
print(f"Màxim error: {np.max(errors):.2f}°")


# ## Plots resultat model 1

# In[11]:


import matplotlib.pyplot as plt

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for images, targets in test_loader:
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
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


# In[12]:


errors = np.abs(all_preds - all_targets)

plt.figure(figsize=(8, 4))
plt.hist(errors, bins=50, color='skyblue', edgecolor='black')
plt.xlabel("Error absoluto (°)")
plt.ylabel("Frecuencia")
plt.title("Distribución del error absoluto")
plt.grid(True)
plt.show()


# In[13]:


plt.figure(figsize=(8, 4))
plt.scatter(all_targets, errors, alpha=0.3, s=10)
plt.xlabel("Ángulo real (°)")
plt.ylabel("Error absoluto (°)")
plt.title("Error por ángulo")
plt.grid(True)
plt.show()


# In[14]:


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

