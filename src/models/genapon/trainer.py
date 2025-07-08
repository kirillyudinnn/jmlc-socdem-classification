import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from structlog import get_logger

from datetime import datetime

from clearml import Task

class Trainer:
    """
        Trainer class for GENAPON model

    Args:
        device (str): training device
        verbose (bool): whether to verbose
    """
    def __init__(self, device=None, verbose=True, clearml=False):
        self.device = device
        self.verbose = verbose
        self.clearml = clearml
        if verbose:
            self.logger = get_logger()

        if self.clearml:
            self.task = Task.init(project_name="Socdem Model", task_name=f"Training GENAPON {datetime.now()}")
            self.task_logger = self.task.get_logger()
        
        self.reset_history()

    def close_task(self):
        if self.clearml:
            self.task.close()
        
    def reset_history(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_f1': [],
            'val_f1': [],
            'train_auc': [],
            'val_auc': [],
            "thresholds" : []
        }
    
    def _compute_metrics(self, logits, y_true, threshold = 0.0):
        """
            Compute F1-score and ROC-AUC

        Args:
            logits (torch.tensor): logits from classification layer's output
            y_true (torch.tensor): batch's users true target
        """
        y_proba = torch.sigmoid(logits).cpu().numpy()
        y_true = y_true.cpu().numpy()

        auc = roc_auc_score(y_true, y_proba)

        if np.isclose(threshold, 0.0):
            thresholds = np.linspace(0, 1, 30)
            best_threshold = 0.0
            best_f1 = 0.0

            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                current_f1 = f1_score(y_true, y_pred)

                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_threshold = threshold
        else:
            y_pred = (y_proba >= threshold).astype(int)
            best_f1 = f1_score(y_true, y_pred)
            best_threshold = threshold
        
        return best_f1, auc, best_threshold
    
    def _train_epoch(self, model, loader, criterion, optimizer):
        """
            Train one epoch

        Args:
            model (GENAPON): model
            loader (DataLoader): train dataloader
            criterion (): Loss function
            optimizer (): Optimizer
        """
        model.train()
        epoch_loss = 0
        all_logits, all_labels = [], []
        
        for X_batch in tqdm(loader, desc="Training", disable=not self.verbose):
            y_batch = X_batch["target"].to(self.device).float()
            
            optimizer.zero_grad()
            logits = model(X_batch).squeeze()
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            all_logits.append(logits.detach())
            all_labels.append(y_batch.detach())
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        f1, auc, best_threshold = self._compute_metrics(all_logits, all_labels)
        return epoch_loss / len(loader), f1, auc, best_threshold
    
    def _validate(self, model, loader, criterion):
        """
            Model validation

        Args:
            model (GENAPON): model
            loader (DataLoader): val dataloader
            criterion (): Loss function
        """
        model.eval()
        val_loss = 0
        all_logits, all_labels = [], []
        
        with torch.no_grad():
            for X_batch in tqdm(loader, desc="Validation", disable=not self.verbose):
                y_batch = X_batch["target"].to(self.device).float()
                
                logits = model(X_batch).squeeze()
                loss = criterion(logits, y_batch)
                
                val_loss += loss.item()
                all_logits.append(logits)
                all_labels.append(y_batch)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        f1, auc, best_threshold = self._compute_metrics(all_logits, all_labels)
        return val_loss / len(loader), f1, auc, best_threshold
    
    def fit(
        self,
        model,
        train_loader,
        val_loader,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=None,
        epochs=10,
        patience=3,
        lr=1e-3,
        scheduler=None
    ):
        """
            model (GENAPON): model
            train_loader (DataLoader): Train dataloader
            val_loader (DataLoader): Val dataloader
            criterion (): Loss function
            optimizer (): Optimizer (Adam by default)
            scheduler (): Scheduler for learning_rate 
            lr (float): start learning rate
            epochs (int): number of training epochs
            patience (int): scheduler's patience
        """
        model = model.to(self.device)
        optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=patience//2
        )

        if self.clearml:
            self.task.connect(
                {
                    "epochs" : epochs,
                    "patience" : patience,
                    "lr" : lr,
                    "batch_size" : train_loader.batch_size,
                    "device" : self.device 
                }
            )
        
        self.reset_history()
        best_val_auc = 0
        no_improve = 0
        
        for epoch in range(1, epochs+1):
            if self.verbose:
                self.logger.info(f"\nEpoch {epoch}/{epochs}")
                self.logger.info("-" * 30)
            
            train_loss, train_f1, train_auc, _ = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_f1, val_auc, best_threshold = self._validate(model, val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            self.history["thresholds"].append(best_threshold)


            if self.clearml:
                self.task_logger.report_scalar("Loss", "Train", value=train_loss, iteration=epoch)
                self.task_logger.report_scalar("Loss", "Validation", value=val_loss, iteration=epoch)
                self.task_logger.report_scalar("F1", "Train", value=train_f1, iteration=epoch)
                self.task_logger.report_scalar("F1", "Validation", value=val_f1, iteration=epoch)
                self.task_logger.report_scalar("ROC-AUC", "Train", value=train_auc, iteration=epoch)
                self.task_logger.report_scalar("ROC-AUC", "Validation", value=val_auc, iteration=epoch)
                self.task_logger.report_scalar("LR", "Value", value=optimizer.param_groups[0]['lr'], iteration=epoch)
                self.task_logger.report_scalar("Threshold", "Value", value=best_threshold, iteration=epoch)

            
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
                best_weights = model.state_dict()
            else:
                no_improve += 1
            
            if self.verbose:
                self.logger.info(
            f'''
                Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}
                Train F1 score: {train_f1:.4f} | Val F1 score: {val_f1:.4f}
                Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}
                Learning rate: {optimizer.param_groups[0]['lr']:.2e}
            '''
            )
            
            if no_improve >= patience:
                if self.verbose:
                    self.logger.info(f"\nEarly stopping at epoch {epoch}")
                model.load_state_dict(best_weights)
                break

        self.task.close()
        return self.history

    def eval_model(self, model, loader, threshold):
        model.eval()
        all_logits, all_labels = [], []
        
        with torch.no_grad():
            for X_batch in tqdm(loader, desc="Evaluating", disable=not self.verbose):
                y_batch = X_batch["target"].to(self.device).float()
                
                logits = model(X_batch).squeeze()

                all_logits.append(logits)
                all_labels.append(y_batch)
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        f1, auc, best_threshold = self._compute_metrics(all_logits, all_labels, threshold)
        return f1, auc, best_threshold
