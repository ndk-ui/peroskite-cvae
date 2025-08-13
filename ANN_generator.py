import numpy as np 
import pandas as pd 
import os, ast
import torch 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from network import DNN 

from sklearn.model_selection import train_test_split
from tqdm import tqdm 

class generatorModel():
    def __init__(self, sizes, latent_size, act, drop_frac):
        df = pd.read_csv('data/processed_data.csv')
        df = self.normalize_data(df)
        
        train_df, test_df = train_test_split(df, test_size = 0.2, random_state=42)
        self.train_X, self.train_Y = self.target_feature(train_df)
        self.test_X, self.test_Y = self.target_feature(test_df)
        
        self.train_loss = []
        self.val_loss = []
        self.lr = []
        
        encoder_sizes = [self.train_X.shape[-1]+self.train_Y.shape[-1]]+sizes+[2*latent_size]
        decoder_sizes = [latent_size+self.train_Y.shape[-1]]+sizes[::-1]+[self.train_X.shape[-1]]
        self.encoder = DNN(encoder_sizes, act, drop_frac)
        self.decoder = DNN(decoder_sizes, act, drop_frac)
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=1e-3,
            weight_decay=1e-3
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.5, 
            patience=50
        )
    
    def normalize_data(self, df):
        df = df.copy()
        df['encoded_data'] = df['encoded_data'].apply(ast.literal_eval)

        y = df.iloc[:, 2:6].values
        self.Y_mean = y.mean(axis = 0)
        self.Y_std = y.std(axis = 0) 
        df.iloc[:,2:6] = (y-self.Y_mean)/self.Y_std
        return df
        
    def target_feature(self, df):
        X = torch.tensor(df['encoded_data'].tolist(), dtype = torch.float32)
        Y = torch.tensor(df.iloc[:, 2:6].values, dtype = torch.float32)
        return X,Y
    
    def latent_sampling(self, mu, log_sigma):
        std = torch.exp(0.5*log_sigma)
        eps = torch.randn_like(std)
        lat  = mu+eps*std
        return lat
    
    def criterions(self, x_true, x_pred, mu,log_sigma, recon_weight = 1, kl_weight = 0.01):
        x_true = x_true.view(-1,23,3)
        x_pred = x_pred.view(-1,23,3)
        x_max = torch.argmax(x_true, dim = 1)
        recon = recon_weight*torch.nn.functional.cross_entropy(x_pred, x_max, reduction='mean')
        
        kl = kl_weight*torch.sum(
            torch.exp(log_sigma)+mu**2-1-log_sigma, dim = 1
        ).mean()
        
        return recon+kl
    
    def train_step(self, X,Y, recon_weight = 1, kl_weight = 0.01):
        self.decoder.train()
        self.encoder.train()
        self.optimizer.zero_grad()
        
        result = self.encoder(torch.cat([X,Y], dim = 1))
        mu, log_sigma = torch.chunk(result, 2, dim = 1)
        Z = self.latent_sampling(mu, log_sigma)
        X_pred = self.decoder(torch.cat([Z,Y], dim = 1))
        
        loss = self.criterions(X,X_pred, mu, log_sigma, recon_weight, kl_weight)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def val_step(self, X,Y, recon_weight = 1, kl_weight = 0.01):
        self.decoder.eval()
        self.encoder.eval()
        with torch.no_grad():
            result = self.encoder(torch.cat([X,Y], dim = 1))
            mu, log_sigma = torch.chunk(result, 2, dim = 1)
            Z = self.latent_sampling(mu, log_sigma)
            X_pred = self.decoder(torch.cat([Z,Y], dim = 1))

            loss = self.criterions(X,X_pred, mu, log_sigma, recon_weight, kl_weight)

        return loss.item()
    
    def train_model(self, epochs, batchsize=64, recon_weight = 1, kl_weight = 0.01):
        train_dataset = TensorDataset(self.train_X, self.train_Y)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle = True)
        
        test_dataset = TensorDataset(self.test_X, self.test_Y)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        
        bar = tqdm(range(epochs+1), desc = "Training")
        for epoch in bar: 
            train_loss = 0 
            val_loss = 0 
            
            for x_batch, y_batch in train_loader:
                batch_loss = self.train_step(x_batch, y_batch, recon_weight = recon_weight, kl_weight = kl_weight)
                train_loss += batch_loss
            for x_batch, y_batch in test_loader:
                batch_loss = self.val_step(x_batch, y_batch, recon_weight = recon_weight, kl_weight = kl_weight)
                val_loss += batch_loss
            
            train_loss/=len(train_loader)
            val_loss /= len(test_loader)
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]['lr']
            
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            self.lr.append(lr)
            
            bar.set_postfix({
            "epoch": epoch,
            "train_loss": self.train_loss[-1],
            "val_loss": self.val_loss[-1],
            'lr': self.lr[-1]
            })
        return self.val_loss[-1]
    def save_model(self, dir_path = '.', filepath = 'model.pth'):
        checkpoint = {
            'train_loss': self.train_loss, 
            'val_loss': self.val_loss, 
            'decoder_state_dict': self.decoder.state_dict(),
            'encoder_state_dict': self.encoder.state_dict(),
            'y_mean': self.Y_mean,
            'y_std': self.Y_std
        }
        filepath = os.path.join(dir_path, filepath)
        os.makedirs(dir_path, exist_ok = True)
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, weights_only=False)
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.Y_mean = checkpoint['y_mean']
        self.Y_std = checkpoint['y_std']
        self.train_loss = checkpoint['train_loss']
        self.val_loss = checkpoint['val_loss']
    
    def inference(self, param, latent_vector):
        param = (param-self.Y_mean)/self.Y_std
        param = torch.Tensor(param)
        latent_vector = torch.Tensor(latent_vector)
        with torch.no_grad():
            X = self.decoder(torch.cat([latent_vector, param], dim = 1))
            X = torch.nn.Softmax(dim=1)(X.view(-1, 23, 3))
        # Convert to one-hot
        pred_classes = torch.argmax(X, dim=1)  # [batch, positions]
        binary_output = torch.nn.functional.one_hot(pred_classes, num_classes=23) \
                          .permute(0, 2, 1).float()
        
        return binary_output
    
    
    