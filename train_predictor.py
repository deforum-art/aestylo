import pathlib
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange


class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            #nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train_predictor(root_folder, train_from, clip_model="ViT-L/14", val_percentage=0.05, epochs=50, batch_size=256):
    
    out_path = pathlib.Path(root_folder)
    x_out = f"x_{clip_model.replace('/', '').lower()}_ebeddings.npy"
    y_out = f"y_{train_from}.npy"
    save_name = f"linear_predictor_{clip_model.replace('/', '').lower()}_{train_from}_mse.pth"
    
    x = np.load(out_path / x_out)
    y = np.load(out_path / y_out)

    train_border = int(x.shape[0] * (1 - val_percentage))

    train_tensor_x = torch.Tensor(x[:train_border])
    train_tensor_y = torch.Tensor(y[:train_border])
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    val_tensor_x = torch.Tensor(x[train_border:])
    val_tensor_y = torch.Tensor(y[train_border:])
    val_dataset = TensorDataset(val_tensor_x, val_tensor_y)
    val_loader = DataLoader(val_dataset, batch_size=512, num_workers=16)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if clip_model in ["ViT-L/14"]:
        model = MLP(768).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    criterion = nn.MSELoss()
    criterion2 = nn.L1Loss()

    model.train()
    best_loss = 999

    for epoch in trange(epochs):
        losses = []
        losses2 = []
        for batch_num, input_data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())

            optimizer.step()

            if batch_num % 1000 == 0:
                pass
                #print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))

        #print('Epoch %d | Loss %6.2f' % (epoch, sum(losses) / len(losses)))
        losses = []
        losses2 = []

        for batch_num, input_data in enumerate(val_loader):
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)

            output = model(x)
            loss = criterion(output, y)
            lossMAE = criterion2(output, y)
            #loss.backward()
            losses.append(loss.item())
            losses2.append(lossMAE.item())
            #optimizer.step()

            if batch_num % 1000 == 0:
                pass
                #print('\tValidation - Epoch %d | Batch %d | MSE Loss %6.2f' % (epoch, batch_num, loss.item()))
                #print('\tValidation - Epoch %d | Batch %d | MAE Loss %6.2f' % (epoch, batch_num, lossMAE.item()))
                
                #print(y)

        #print('Validation - Epoch %d | MSE Loss %6.2f' % (epoch, sum(losses)/len(losses)))
        #print('Validation - Epoch %d | MAE Loss %6.2f' % (epoch, sum(losses2)/len(losses2)))
        if sum(losses)/len(losses) < best_loss:
            #print("Best MAE Val loss so far. Saving model")
            best_loss = sum(losses)/len(losses)
            #print( best_loss ) 

            torch.save(model.state_dict(), out_path / save_name )

    torch.save(model.state_dict(), out_path /  save_name)

    #print(best_loss) 
    #print("training done")
    # inferece test with dummy samples from the val set, sanity check
    #print( "inferece test with dummy samples from the val set, sanity check")
    model.eval()
    output = model(x[:5].to(device))
    #print(output.size())
    #print(output)