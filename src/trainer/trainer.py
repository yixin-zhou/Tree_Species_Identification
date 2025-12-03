import os
import csv
import torch
import logging
from tqdm import tqdm

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
        self.loss_log_path = os.path.join(self.best_model_path, "losses.csv")
        
        os.makedirs(self.best_model_path, exist_ok=True)

        self.wandb_run = wandb_run
        
        # 提前写入表头，方便追踪 loss 曲线
        if not os.path.exists(self.loss_log_path):
            with open(self.loss_log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss"])

    def _move_to_device(self, inputs, targets):
        inputs_device = {k: v.to(self.device) for k, v in inputs.items()}
        
        targets_device = []
        for t in targets:
            t_device = {
                'boxes': t['boxes'].to(self.device),
                'labels': t['labels'].to(self.device)
            }
            targets_device.append(t_device)
            
        return inputs_device, targets_device

    def train_one_epoch(self, train_loader, current_epoch_num):
        self.model.train()
        total_loss_epoch = 0.0
        
        total_cls_loss = 0.0
        total_bbox_loss = 0.0
        total_ctr_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training Epoch {current_epoch_num}")
        
        for inputs, targets in pbar:
            inputs, targets = self._move_to_device(inputs, targets)
            
            self.optimizer.zero_grad()

            outputs = self.model(inputs)

            loss_dict = self.criterion(outputs, targets)

            loss = loss_dict["loss_cls"] + loss_dict["loss_bbox"] + loss_dict["loss_centerness"]
            
            loss.backward()
            self.optimizer.step()

            total_loss_epoch += loss.item()
            total_cls_loss += loss_dict["loss_cls"].item()
            total_bbox_loss += loss_dict["loss_bbox"].item()
            total_ctr_loss += loss_dict["loss_centerness"].item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss_epoch / len(train_loader)
        
        if self.wandb_run is not None:
            self.wandb_run.log({
                "epoch": current_epoch_num,
                "train/total_loss": avg_loss,
                "train/loss_cls": total_cls_loss / len(train_loader),
                "train/loss_bbox": total_bbox_loss / len(train_loader),
                "train/loss_ctr": total_ctr_loss / len(train_loader),
            })
            
        return avg_loss
    
    def validate(self, val_loader, current_epoch_num):
        self.model.eval()
        total_loss_epoch = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validating Epoch {current_epoch_num}"):
                inputs, targets = self._move_to_device(inputs, targets)

                outputs = self.model(inputs)
                loss_dict = self.criterion(outputs, targets)
                
                loss = loss_dict["loss_cls"] + loss_dict["loss_bbox"] + loss_dict["loss_centerness"]
                total_loss_epoch += loss.item()

        avg_loss = total_loss_epoch / len(val_loader)
        
        return avg_loss

    def _record_losses(self, epoch, train_loss, val_loss=None):
        with open(self.loss_log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, "" if val_loss is None else val_loss])
    
    def fit(self, train_loader, epochs, val_loader=None):
        logger.info(f"Start Training for {epochs} epochs...")

        for epoch in range(1, epochs+1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            self.train_loss.append(train_loss)
            
            log_msg = f"Epoch: {epoch} | Train Loss: {train_loss:.4f}"

            if val_loader is not None:
                val_loss = self.validate(val_loader, epoch)
                self.val_loss.append(val_loss)
                
                log_msg += f" | Val Loss: {val_loss:.4f}"
                
                if self.wandb_run is not None:
                    self.wandb_run.log({
                        "epoch": epoch,
                        "Validation Loss": val_loss,
                        "Train Loss": train_loss
                    })
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    save_path = os.path.join(self.best_model_path, 'best_model.pth')
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"New best model saved to {save_path}")

                self._record_losses(epoch, train_loss, val_loss)
            logger.info(log_msg)

        # Save last model
        last_path = os.path.join(self.best_model_path, 'last_model.pth')
        torch.save(self.model.state_dict(), last_path)
        logger.info(f"Training Finished. Last model saved to {last_path}")
