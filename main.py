
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
from model import NGP
from utils import train, test

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = torch.from_numpy(np.load('training_data_800x800.pkl',
                                             allow_pickle=True))
    testing_dataset = torch.from_numpy(np.load('testing_data_800x800.pkl', 
                                               allow_pickle=True))

    L = 16
    F = 2
    T = 2**19
    N_min = 16
    N_max = 2048
    b = np.exp((np.log(N_max) - np.log(N_min) / (L - 1)))
    Nl = [int(np.floor(N_min * b**l)) for l in range(L)]
    model = NGP(T, Nl, 4, device, 3)
    
    model_optimizer = torch.optim.Adam(
        [{"params": model.lookup_tables.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 0.},
        {"params": model.density_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10**-6},
        {"params": model.color_MLP.parameters(), "lr": 1e-2, "betas": (0.9, 0.99), "eps": 1e-15, "weight_decay": 10**-6},
        ])

    data_loader = DataLoader(train_dataset, batch_size=2**14, shuffle=True)
    print(f"[INFO] starting training phase")
    train(model, model_optimizer, data_loader, nb_epochs=1, device=device,
          hn=2,hf=6, nb_bins=192, H=800, W=800)
    
    print(f"[INFO] Finished training...")
    
    print(f"[INFO] starting testing phase")
    progress_bar = tqdm(range(200), total=200)
    for img_index in progress_bar:
        progress_bar.set_description(f"image: {img_index}")
        test(model=model, device=device, hn=2, hf=6, dataset=testing_dataset, img_index=img_index, nb_bins=192, H=800, W=800)
    print(f"[INFO] finished testing phase, outputs in novel_views/.")    
    
    