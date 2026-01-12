# src/training/trainer.py
import torch
from tqdm import tqdm
import numpy as np

def diag_mat_weights(dimp, type='first'):
    """
    Return a torch Tensor D matrix:
    - 'first': (dimp-1, dimp) with [-1, 1] on diag and next diag
    - 'second': (dimp-2, dimp) second difference [-1,2,-1]
    """
    if type == 'first':
        dg = np.zeros((dimp-1, dimp))
        for i in range(dimp-1):
            dg[i,i] = -1
            dg[i,i+1] = 1
    elif type == 'second':
        dg = np.zeros((dimp-2, dimp))
        for i in range(dimp-2):
            dg[i,i] = -1
            dg[i,i+1] = 2
            dg[i,i+2] = -1
    else:
        raise ValueError("Unknown type")
    return torch.tensor(dg, dtype=torch.float32)




class Trainer:
    def __init__(self, model, optimizer, criterion, device, scheduler=None, early_stopping=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, train_loader, penalty_func=None, lambda_vals=None):
        
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 1. Forward Pass
            output = self.model(batch_x)
            loss = self.criterion(output, batch_y)
            

            # 2. Add Penalty
            if penalty_func:
                reg_loss = penalty_func(self.model, lambda_vals, self.device)
                loss += (reg_loss)
            
            # 3. Backward Pass
            loss.backward()
            
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_loader)
        self.history['train_loss'].append(epoch_loss)
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                running_loss += loss.item()
                
        epoch_loss = running_loss / len(val_loader)
        self.history['val_loss'].append(epoch_loss)
        
        # Early Stopping Check
        if self.early_stopping:
            self.early_stopping(epoch_loss, self.model)
        
        return epoch_loss

    def fit(self, train_loader, val_loader, epochs, penalty_func=None, lambda_vals=None):
        pbar = tqdm(range(epochs), desc="Epochs")
        for epoch in pbar:
            train_loss = self.train_epoch(train_loader, penalty_func, lambda_vals)
            val_loss = self.validate(val_loader)
            
            if self.scheduler:
                self.scheduler.step()
                
            pbar.set_postfix({'Train': f"{train_loss:.4f}", 'Val': f"{val_loss:.4f}"})
            
            if self.early_stopping and self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        return self.history