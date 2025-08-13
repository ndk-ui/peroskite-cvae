from ANN_predictor import predictorModel
import numpy as np 
import pandas as pd 
import torch
import os
import matplotlib.pyplot as plt

#Category : 'Bandgap', 'volumepa', 'Atom_energy', 'Density'
sizes = [512, 256, 64, 32] #only hidden layer
act = torch.nn.ReLU
drop_frac = 0.35
category = 'Bandgap'

predictor = predictorModel(sizes, act, drop_frac, category)
predictor.train_model(700,64)
predictor.save_model('model/bandgap')
predictor.load_model('model/bandgap/model.pth')
df = pd.read_csv('data/processed_data.csv')
predictor.normalize_data(df, category)
X,Y = predictor.target_feature(df, category)
Y_pred = predictor.inference(X)
r2 = predictor.r2_score(Y_pred, Y)


Y_true = predictor.denormalize(Y).detach().numpy()
Y_pred = predictor.denormalize(Y_pred).detach().numpy()

# Scatter plot of predictions vs actual
plt.scatter(Y_true, Y_pred, alpha=0.6, label='Predictions')

# Perfect prediction line (y = x)
min_val = min(Y_true.min(), Y_pred.min())
max_val = max(Y_true.max(), Y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label=f'Perfect fit R2 = {r2:.2f}')

plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title('Bandgap Plot')
plt.grid()
plt.legend()
os.makedirs('figure', exist_ok=True)
plt.savefig('figure/Bandgap.png')
