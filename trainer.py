import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utils.utils import plot_training_curves

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001,  
                         use_mixup=True, label_smoothing=0.1, warmup_epochs=5, 
                         patience=10, device=None, model_save_path='best_color_model.pth'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    print(f"Training on device: {device}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    backbone_params = []
    color_params = []
    
    for name, param in model.named_parameters():
        if 'color' in name.lower() or 'histogram' in name.lower():
            color_params.append(param)
        else:
            backbone_params.append(param)
    
    if color_params:
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': lr, 'weight_decay': 0.01},
            {'params': color_params, 'lr': lr * 2, 'weight_decay': 0.005}
        ], lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    
    def mixup_data(x, y, alpha=1.0):
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample().item() 
        else:
            lam = 1.0 
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    best_val_acc = 0.0
    patience_counter = 0
    
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            if use_mixup and torch.rand(1).item() > 0.5: 
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=0.2)
                outputs = model(inputs)
                loss = mixup_criterion(outputs, targets_a, targets_b, lam)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_train += targets_a.size(0)
                correct_train += (lam * predicted.eq(targets_a).sum().item() +
                                (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_train += targets.size(0)
                correct_train += predicted.eq(targets).sum().item()
        
        epoch_train_loss = running_loss / total_train if total_train > 0 else 0
        epoch_train_acc = 100. * correct_train / total_train if total_train > 0 else 0
        
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total_val += targets.size(0)
                correct_val += predicted.eq(targets).sum().item()
        
        epoch_val_loss = running_val_loss / total_val if total_val > 0 else 0
        epoch_val_acc = 100. * correct_val / total_val if total_val > 0 else 0
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch+1}/{num_epochs}, LR: {current_lr:.6f}, '
              f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% (Best: {best_val_acc:.2f}%)')
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with Val Acc: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation accuracy for {patience} epochs.")
            break
            
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded best model from {model_save_path} for final use.")
    
    plot_training_curves(train_losses, train_accs, val_losses, val_accs) 
    
    return model

def evaluate_model(model, test_loader, class_names, device=None, conf_matrix_path='confusion_matrix.png'):
    """Evaluate model with detailed metrics"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    accuracy = np.mean(all_preds == all_targets) * 100
    
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    unique_labels = np.unique(np.concatenate((all_targets, all_preds)))
    current_class_names = [class_names[i] for i in unique_labels if i < len(class_names)]

    if not current_class_names or len(current_class_names) != len(unique_labels):
        print(f"Warning: class_names ({class_names}) might not align with detected labels ({unique_labels}). Using default numeric labels for report.")
        report = classification_report(all_targets, all_preds, zero_division=0)
    else:
        report = classification_report(all_targets, all_preds, target_names=current_class_names, zero_division=0)
    print(report)

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=current_class_names if current_class_names and len(current_class_names) == cm.shape[1] else False, 
                yticklabels=current_class_names if current_class_names and len(current_class_names) == cm.shape[0] else False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(conf_matrix_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to {conf_matrix_path}")
    
    return accuracy, all_preds, all_targets, all_probs