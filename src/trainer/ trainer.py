import os
import torch
import logging
from tqdm import tqdm
from torch.utils.data import ConcatDataset

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model, criterion, optimizer, device, pth_savepath, wandb_run=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

        self.train_loss = []
        self.val_loss = []
        self.best_val_loss = float("inf")
        self.best_model_path = pth_savepath

        self.wandb_run = wandb_run

    def train_one_epoch(self, train_loader, current_epoch_num):
        self.model.train()
        total_loss = 0.0

        for modalities, mask, labels in tqdm(train_loader, desc=f"Training Epoch {current_epoch_num}"):
            modalities = modalities.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(modalities)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader, current_epoch_num):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for modalities, mask, labels in tqdm(val_loader, desc=f"Validating Epoch {current_epoch_num}"):
                modalities = modalities.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(modalities)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(val_loader)
        acc = correct / total
        return avg_loss, acc
    
    def fit(self, train_loader, epochs, val_loader=None):
        logger.info(f"Start Training for {epochs} epochs...")

        for epoch in range(1, epochs+1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            self.train_loss.append(train_loss)
            
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader, epoch)
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_acc": val_acc
                    })
                logger.info(f"Epoch: {epoch} train_loss: {train_loss} val_loss: {val_loss} val_acc: {val_acc}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.best_model_path, 'best_model.pth'))
                    logger.info(f"New best model saved to {os.path.join(self.best_model_path, 'best_model.pth')}")
                    self.val_loss.append(val_loss)

            else:
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "epoch": epoch,
                        "train_loss": train_loss,
                    })
                logger.info(f"Epoch: {epoch} train_loss: {train_loss}")

        
        torch.save(self.model.state_dict(), os.path.join(self.best_model_path, 'last_model.pth'))
        logger.info(f"Training Finished. Last model saved to {os.path.join(self.best_model_path, 'last_model.pth')}")




        