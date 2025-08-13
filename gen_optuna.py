from ANN_generator import generatorModel
import torch, optuna
from torch.utils.data import TensorDataset, DataLoader

def objective(trial):
    sizes_options = [
        [256,128,64,32],
        [128,64,32,16],
        [512,256,128,64],
        [128,128,64,32],
        [256,128,128,64]
    ]
    act_options = [torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.Tanh]
    sizes = trial.suggest_categorical("sizes", sizes_options)
    latent_size = trial.suggest_int("latent_size", 4, 32, step = 4)
    act = trial.suggest_categorical("activation", act_options) 
    drop_frac = trial.suggest_float("drop_frac", 0.0, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [32,64,128])
    recon_weight = trial.suggest_float("recon_weight", 1.0, 5.0)
    kl_weight = trial.suggest_float("kl_weight", 0.01, 0.5)
    
    generator = generatorModel(sizes, latent_size, act, drop_frac)
    val_loss = generator.train_model(200, batchsize=batch_size, recon_weight=recon_weight, kl_weight=kl_weight)

    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials = 500)

print("Best parameters:", study.best_params)
with open("best_params.txt", "w") as f:
    f.write("Best parameters:\n")
    for key, value in study.best_params.items():
        f.write(f"{key}: {value}\n")
    