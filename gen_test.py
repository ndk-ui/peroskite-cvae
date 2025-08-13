from ANN_generator import generatorModel
import numpy as np 
import pandas as pd 
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import data_processing as dp

with open("data/unique_char.txt", "r") as f:
    unique_char = f.read().splitlines() 
    
#Category : 'Bandgap', 'volumepa', 'Atom_energy', 'Density'
sizes = [256, 128, 64, 32] #only hidden layer
latent_size = 8
act = torch.nn.ReLU
drop_frac = 0.35
batch_size = 64
recon_weight = 1
kl_weight = 1

generator = generatorModel(sizes, latent_size, act, drop_frac)
train_dataset = TensorDataset(generator.train_X, generator.train_Y)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle = True)

val_loss = generator.train_model(1000,batch_size, recon_weight, kl_weight)
generator.save_model('model/generator')
generator.load_model('model/generator/model.pth')

df = pd.read_csv('data/processed_data.csv')
param = df.iloc[:, 2:6].values
latent_vector = np.random.rand(len(param), latent_size)

X_pred = generator.inference(param, latent_vector).numpy()
df['predicted_data'] = X_pred.tolist()
df['predicted_comp'] = df['predicted_data'].apply(lambda x: dp.backprocess_one_hot(np.array(x), unique_char))
df.to_csv('data/predicted_data.csv')

plt.plot(generator.train_loss, 'k-', label='Train Loss')
plt.plot(generator.val_loss, 'r--', label='Validation Loss')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.title('Loss History')
plt.legend()

ax = plt.gca()
plt.savefig('figure/Generator.png')
plt.show()
