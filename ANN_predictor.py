import numpy as np 
import pandas as pd 
import os, ast
import torch 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from network import DNN 

from sklearn.model_selection import train_test_split
from tqdm import tqdm 

#Category : 'Bandgap', 'volumepa', 'Atom_energy', 'Density'
class predictorModel():
    def __init__(self, sizes, act, drop_frac, category):      
        df = pd.read_csv('data/processed_data.csv')
        self.normalize_data(df, category)
        
        train_df, test_df = train_test_split(df, test_size = 0.2, random_state=42)
        self.X_train, self.Y_train = self.target_feature(train_df, category)
        self.X_test, self.Y_test = self.target_feature(test_df, category)
  
        self.train_loss = []
        self.val_loss = []
        self.train_r2 = []
        self.val_r2 = []
        self.lr = []

        sizes.insert(0, self.X_train.shape[-1])
        sizes.append(self.Y_train.shape[-1])
        self.DNN = DNN(sizes, act, drop_frac)    
        self.optimizer = torch.optim.Adam(self.DNN.parameters(), lr=1e-3, weight_decay=1e-3)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode = 'max',
            factor = 0.5, 
            patience = 100,
        )
    
    def normalize_data(self, df, category):
        df['encoded_data'] = df['encoded_data'].apply(ast.literal_eval)
        
        y = df[category].values
        self.Y_mean = y.mean()
        self.Y_std = y.std() 
        df[category] = (y-self.Y_mean)/self.Y_std
        
    def target_feature(self, df, category):
        X = torch.tensor(df['encoded_data'].tolist(), dtype = torch.float32)
        Y = torch.tensor(df[category].values, dtype = torch.float32).reshape(-1,1)
        return (X,Y)
    
    def criterion(self, prediction, target):
        return torch.nn.functional.mse_loss(prediction, target, reduction='mean')
    
    def r2_score(self, prediction, target):
        #1 - (sum((y_true - y_pred)^2) / sum((y_true - y_mean)^2))
        prediction = self.denormalize(prediction)
        target = self.denormalize(target)
        ss_res = torch.sum((target-prediction)**2)
        ss_tot = torch.sum((target-torch.mean(target))**2)
        r2 = 1-ss_res/ss_tot 
        return r2
    
    def train_step(self, X, Y):
        self.DNN.train()
        self.optimizer.zero_grad()
        Y_pred = self.DNN(X)
        loss = self.criterion(Y_pred, Y)
        loss.backward() 
        self.optimizer.step()   
        
        with torch.no_grad():
            r2 = self.r2_score(Y_pred, Y)
                
        return loss.item(), r2.item()
    
    def val_step(self, X, Y):
        self.DNN.eval()
        Y_pred = self.DNN(X)
        loss = self.criterion(Y_pred, Y)
        r2 = self.r2_score(Y_pred, Y)
        
        return loss.item(), r2.item()
    
    def train_model(self, epochs, batchsize = 64):
        train_dataset = TensorDataset(self.X_train, self.Y_train)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle = True)
        
        test_dataset = TensorDataset(self.X_test, self.Y_test)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        bar = tqdm(range(epochs+1), desc = "Training")
        for epoch in bar:
            #Training
            train_loss = 0 
            train_r2 = 0 
            for x_batch, y_batch in train_loader:
                batch_loss, batch_r2 = self.train_step(x_batch, y_batch)
                train_loss+=batch_loss
                train_r2+=batch_r2
            #Validation 
            val_loss = 0 
            val_r2 = 0 
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    batch_loss, batch_r2 = self.val_step(x_batch, y_batch)
                    val_loss+=batch_loss
                    val_r2+=batch_r2
            #Avg metrics
            train_loss/=len(train_loader)
            train_r2/=len(train_loader)
            val_loss/=len(test_loader)
            val_r2/=len(test_loader)
            #Append history 
            self.train_loss.append(train_loss)
            self.train_r2.append(train_r2)
            self.val_loss.append(val_loss)
            self.val_r2.append(val_r2)
            #Scheduler step
            self.scheduler.step(val_r2)
            self.lr.append(self.optimizer.param_groups[0]['lr'])
            #Progress bar
            bar.set_postfix({
            "epoch": epoch,
            "train_loss": self.train_loss[-1],
            "val_loss": self.val_loss[-1],
            "train_r2": self.train_r2[-1],
            "val_r2": self.val_r2[-1],
            'lr': self.lr[-1]
        })
            
    def save_model(self, dir_path = '.', filepath = 'model.pth'):
        checkpoint = {
            'train_loss': self.train_loss, 
            'val_loss': self.val_loss, 
            'train_r2': self.train_r2,
            'val_r2': self.val_r2,
            'model_state_dict': self.DNN.state_dict(),
            'y_mean': self.Y_mean,
            'y_std': self.Y_std
        }
        filepath = os.path.join(dir_path, filepath)
        os.makedirs(dir_path, exist_ok = True)
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        self.DNN.load_state_dict(checkpoint['model_state_dict'])
        self.Y_mean = checkpoint['y_mean']
        self.Y_std = checkpoint['y_std']
    
    def denormalize(self, y):
        return y*self.Y_std+self.Y_mean
    
    def inference(self, x):
        with torch.no_grad():
            y = self.DNN(x)
        return y


    