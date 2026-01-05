"""
EEE 443/543 Neural Networks Project
Multi-Class Skin Lesion Classification on ISIC Dataset

This script implements a complete pipeline for training and evaluating three deep learning
models (ResNet18, EfficientNet-B0, and ViT-B/16) on the ISIC skin lesion dataset.

Author: Sefa Emre Kavgacı
Student ID: 22103179
Date: January 2026
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
from PIL import Image
import cv2

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)

class Config:
    # Main project configuration
    
    # Directory setup
    DATA_DIR = "data"
    RESULTS_DIR = "results"
    
    # Fix seed for reproducibility
    RANDOM_SEED = 42
    
    # Standard 80-20 split
    TRAIN_TEST_SPLIT = 0.8
    
    # 5-fold cross-validation
    N_FOLDS = 5
    
    # Training hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 30  # Reduced, let early stopping handle the rest
    EARLY_STOPPING_PATIENCE = 7
    NUM_WORKERS = 2
    
    # Standard ImageNet normalization stats
    IMG_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Adjust learning rates based on architecture
    MODEL_CONFIGS = {
        'resnet18': {
            'lr': 1e-4,
            'weight_decay': 1e-4,
        },
        'efficientnet_b0': {
            'lr': 1e-4,
            'weight_decay': 1e-4,
        },
        'vit_b16': {
            'lr': 5e-5,  # ViT is sensitive, needs lower LR
            'weight_decay': 1e-4,
        }
    }
    
    # All target labels
    CLASS_NAMES = [
        'actinic keratosis',
        'basal cell carcinoma',
        'dermatofibroma',
        'melanoma',
        'nevus',
        'pigmented benign keratosis',
        'seborrheic keratosis',
        'squamous cell carcinoma',
        'vascular lesion'
    ]
    
    # Critical classes for clinical metrics
    MALIGNANT_CLASSES = [
        'actinic keratosis',
        'basal cell carcinoma',
        'melanoma',
        'squamous cell carcinoma'
    ]

def get_device():
    # Pick the best available hardware (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Metal (MPS) GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")
    return device

def set_seed(seed=42):
    # Lock seeds for all libraries to ensure consistent results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collect_image_paths(data_dir):
    # Scrape directory for images and build a dataframe
    data = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for class_name in Config.CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        
        # Skip if folder is missing
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found: {class_dir}")
            continue
        
        # Grab files with valid extensions
        for ext in image_extensions:
            for img_path in Path(class_dir).glob(f'*{ext}'):
                data.append({
                    'image_path': str(img_path),
                    'class_name': class_name,
                    'label': Config.CLASS_NAMES.index(class_name)
                })
            # Handle uppercase extensions just in case
            for img_path in Path(class_dir).glob(f'*{ext.upper()}'):
                data.append({
                    'image_path': str(img_path),
                    'class_name': class_name,
                    'label': Config.CLASS_NAMES.index(class_name)
                })
    
    return pd.DataFrame(data)

def create_train_test_split(df, test_size=0.2, random_state=42):
    # Perform stratified split to maintain class balance across sets
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def save_data_split(train_df, test_df, save_dir):
    # Persist the split data to CSVs to ensure reproducibility
    os.makedirs(save_dir, exist_ok=True)
    
    train_df.to_csv(os.path.join(save_dir, 'train_split.csv'), index=False)
    test_df.to_csv(os.path.join(save_dir, 'test_split.csv'), index=False)
    
    print(f"Data split saved to: {save_dir}")
    print(f"Train: {len(train_df)} samples | Test: {len(test_df)} samples")

def load_data_split(save_dir):
    # Attempt to load existing splits to save time
    train_path = os.path.join(save_dir, 'train_split.csv')
    test_path = os.path.join(save_dir, 'test_split.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return None, None
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Loaded existing data split from {save_dir}")
    print(f"Train: {len(train_df)} samples | Test: {len(test_df)} samples")
    
    return train_df, test_df

def plot_class_distribution(train_df, test_df, save_dir):
    # Generate bar charts to visually verify class distribution
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Training set distribution
    train_counts = train_df['class_name'].value_counts().reindex(Config.CLASS_NAMES)
    axes[0].bar(range(len(train_counts)), train_counts.values, color='steelblue')
    axes[0].set_xticks(range(len(Config.CLASS_NAMES)))
    axes[0].set_xticklabels(Config.CLASS_NAMES, rotation=45, ha='right')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Training Set Distribution')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Test set distribution
    test_counts = test_df['class_name'].value_counts().reindex(Config.CLASS_NAMES)
    axes[1].bar(range(len(test_counts)), test_counts.values, color='coral')
    axes[1].set_xticks(range(len(Config.CLASS_NAMES)))
    axes[1].set_xticklabels(Config.CLASS_NAMES, rotation=45, ha='right')
    axes[1].set_ylabel('Number of Images')
    axes[1].set_title('Test Set Distribution')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================
# DATASET CLASS
# ============================================

class SkinLesionDataset(Dataset):
    # Custom dataset loader handling image paths and transforms
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Handle corrupt images by returning a black placeholder
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================
# DATA TRANSFORMS
# ============================================

def get_train_transforms():
    # Heavy augmentation (flips, rotations, color jitter) to reduce overfitting
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])

def get_val_test_transforms():
    # Standard resizing and normalization for evaluation
    return transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.IMAGENET_MEAN, std=Config.IMAGENET_STD)
    ])

# ============================================
# CLASS WEIGHTS & DATALOADERS
# ============================================

def calculate_class_weights(labels, device):
    # Calculate inverse frequency weights to handle class imbalance
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Standard weighting formula: total / (classes * count)
    class_weights = total_samples / (num_classes * class_counts)
    
    # Normalize weights to keep scale reasonable
    class_weights = class_weights / class_weights.sum() * num_classes
    
    return torch.FloatTensor(class_weights).to(device)

def create_dataloaders(train_df, val_df, batch_size=32, num_workers=2):
    # Setup datasets with appropriate transforms
    train_dataset = SkinLesionDataset(train_df, transform=get_train_transforms())
    val_dataset = SkinLesionDataset(val_df, transform=get_val_test_transforms())
    
    # Train loader needs shuffling, val doesn't
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

# ============================================
# MODEL ARCHITECTURES
# ============================================

def get_resnet18(num_classes=9, pretrained=True):
    # Standard ResNet18, replacing the final FC layer
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model, "resnet18"

def get_efficientnet_b0(num_classes=9, pretrained=True):
    # EfficientNet uses a classifier block, swap the last linear layer
    model = models.efficientnet_b0(pretrained=pretrained)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model, "efficientnet_b0"

def get_vit_b16(num_classes=9, pretrained=True):
    # For ViT, we modify the specific head structure
    model = models.vit_b_16(pretrained=pretrained)
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
    return model, "vit_b16"

def get_model(model_name, num_classes=9, pretrained=True):
    # Factory wrapper to easily switch architectures
    model_dict = {
        'resnet18': get_resnet18,
        'efficientnet_b0': get_efficientnet_b0,
        'vit_b16': get_vit_b16
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Model '{model_name}' not supported. Choose from {list(model_dict.keys())}")
    
    return model_dict[model_name](num_classes, pretrained)

def count_parameters(model):
    # Quick check for model complexity
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def get_model_grad_cam_layer(model_name, model):
    # Identify the target layer for Grad-CAM visualization based on architecture
    if model_name == 'resnet18':
        return model.layer4[-1]  # Last residual block
    elif model_name == 'efficientnet_b0':
        return model.features[-1]  # Last feature extraction layer
    elif model_name == 'vit_b16':
        return model.encoder.layers[-1].ln_1  # Layer norm before final attention
    else:
        raise ValueError(f"Grad-CAM layer not defined for {model_name}")
    
# ============================================
# EARLY STOPPING
# ============================================

class EarlyStopping:
    # Prevents overfitting by monitoring validation loss
    
    def __init__(self, patience=7, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        # Check if current metric is better than best seen so far
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        # Reset counter if improved, otherwise count up towards patience limit
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False

# ============================================
# TRAINING & VALIDATION LOOPS
# ============================================

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    # Run one full training pass, updating weights and tracking loss
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{running_loss / total:.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return running_loss / total, 100. * correct / total

def validate(model, val_loader, criterion, device, epoch):
    # Evaluate model performance on validation set without gradients
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]  ')
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{running_loss / total:.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    return running_loss / total, 100. * correct / total, all_predictions, all_labels

# ============================================
# TRAINING HISTORY
# ============================================

class TrainingHistory:
    # simple class to store metrics and generate plots
    
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []
    
    def update(self, train_loss, train_acc, val_loss, val_acc, lr):
        # Record stats for the current epoch
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)
    
    def save_to_csv(self, save_path):
        # Dump history to CSV for later analysis
        pd.DataFrame({
            'epoch': range(1, len(self.train_losses) + 1),
            'train_loss': self.train_losses,
            'train_acc': self.train_accs,
            'val_loss': self.val_losses,
            'val_acc': self.val_accs,
            'learning_rate': self.learning_rates
        }).to_csv(save_path, index=False)
    
    def plot_curves(self, save_path, title='Training History'):
        # Visualize loss, accuracy, and LR changes over epochs
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curves')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy curves
        axes[1].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2)
        axes[1].plot(epochs, self.val_accs, 'r-', label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate schedule
        axes[2].plot(epochs, self.learning_rates, 'g-', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# ============================================
# SINGLE FOLD TRAINING
# ============================================

def train_single_fold(model_name, train_df, val_df, fold_num, num_epochs, 
                      batch_size, device, save_dir):
    # Orchestrate the full training pipeline for one specific fold
    print(f"\n{'='*60}")
    print(f"Training {model_name.upper()} - Fold {fold_num}")
    print('='*60)
    
    fold_dir = os.path.join(save_dir, model_name, f'fold_{fold_num}')
    os.makedirs(fold_dir, exist_ok=True)
    
    # Initialize model and send to device
    model, _ = get_model(model_name, num_classes=9, pretrained=True)
    model = model.to(device)
    
    # Prepare data loaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, batch_size=batch_size, num_workers=Config.NUM_WORKERS
    )
    
    # Setup loss with class weights to handle imbalance
    train_labels = train_df['label'].values
    class_weights = calculate_class_weights(train_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    model_config = Config.MODEL_CONFIGS[model_name]
    optimizer = optim.Adam(
        model.parameters(),
        lr=model_config['lr'],
        weight_decay=model_config['weight_decay']
    )
    
    # Reduce LR if we hit a plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, mode='max')
    history = TrainingHistory()
    
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\nConfig: {len(train_df)} train | {len(val_df)} val | "
          f"batch={batch_size} | lr={model_config['lr']}\n")
    
    # Main training loop
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device, epoch
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        history.update(train_loss, train_acc, val_loss, val_acc, current_lr)
        
        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  LR: {current_lr:.2e}")
        
        scheduler.step(val_loss)
        
        # Save checkpoint if accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, os.path.join(fold_dir, 'best_model.pth'))
            print(f" Best model saved (Val Acc: {val_acc:.2f}%)")
        
        # Check if we should stop early
        if early_stopping(val_acc, epoch):
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best: {best_val_acc:.2f}% at epoch {best_epoch}")
            break
        
        print()
    
    # Wrap up and save logs/plots
    history.save_to_csv(os.path.join(fold_dir, 'training_log.csv'))
    history.plot_curves(
        os.path.join(fold_dir, 'training_curves.png'),
        title=f'{model_name.upper()} - Fold {fold_num}'
    )
    
    print(f"\n{'='*60}")
    print(f"Fold {fold_num} Complete: Best={best_val_acc:.2f}% @ Epoch {best_epoch}")
    print('='*60 + "\n")
    
    # Clean up memory
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return best_val_acc, best_epoch, history

# ============================================
# CROSS-VALIDATION TRAINING
# ============================================

def train_with_cross_validation(model_name, train_df, n_folds=5, num_epochs=30, 
                                 batch_size=32, device=None):
    # Orchestrate K-Fold CV to validate model robustness
    if device is None:
        device = get_device()
    
    print("\n" + "="*80)
    print(f"CROSS-VALIDATION TRAINING: {model_name.upper()}")
    print("="*80)
    print(f"Samples: {len(train_df)} | Folds: {n_folds} | "
          f"Max Epochs: {num_epochs} | Batch: {batch_size}")
    print("="*80 + "\n")
    
    results_dir = Config.RESULTS_DIR
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.RANDOM_SEED)
    fold_results = []
    
    # Iterate through each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['label']), 1):
        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)
        
        # Train fold and capture metrics
        best_val_acc, best_epoch, history = train_single_fold(
            model_name=model_name,
            train_df=fold_train_df,
            val_df=fold_val_df,
            fold_num=fold,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            save_dir=results_dir
        )
        
        fold_results.append({
            'fold': fold,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_train_loss': history.train_losses[-1],
            'final_val_loss': history.val_losses[-1]
        })
    
    # Aggregate stats across folds
    val_accs = [r['best_val_acc'] for r in fold_results]
    mean_acc = np.mean(val_accs)
    std_acc = np.std(val_accs)
    
    # Report summary
    print("\n" + "="*80)
    print(f"CROSS-VALIDATION SUMMARY: {model_name.upper()}")
    print("="*80)
    print("\nPer-Fold Results:")
    print("-"*80)
    for result in fold_results:
        print(f"  Fold {result['fold']}: {result['best_val_acc']:.2f}% "
              f"(Best Epoch: {result['best_epoch']})")
    print("-"*80)
    print(f"\nStatistics:")
    print(f"  Mean ± Std: {mean_acc:.2f} ± {std_acc:.2f}%")
    print(f"  Range: [{min(val_accs):.2f}%, {max(val_accs):.2f}%]")
    print("="*80 + "\n")
    
    # Serialize results to JSON
    cv_results = {
        'model_name': model_name,
        'n_folds': n_folds,
        'fold_results': fold_results,
        'mean_val_acc': mean_acc,
        'std_val_acc': std_acc,
        'min_val_acc': min(val_accs),
        'max_val_acc': max(val_accs)
    }
    
    summary_path = os.path.join(results_dir, model_name, 'cv_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(cv_results, f, indent=4)
    
    print(f"Results saved to: {summary_path}\n")
    
    return cv_results

# ============================================
# MODEL EVALUATION
# ============================================

def load_best_model(model_name, fold_num, device):
    # Retrieve the saved checkpoint with the highest validation accuracy
    model, _ = get_model(model_name, num_classes=9, pretrained=False)
    
    checkpoint_path = os.path.join(
        Config.RESULTS_DIR, model_name, f'fold_{fold_num}', 'best_model.pth'
    )
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint

def evaluate_on_test_set(model, test_loader, device):
    
    # Run inference on unseen test data to get final metrics
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    correct = 0
    total = 0
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = 100. * correct / total
    return accuracy, all_predictions, all_labels, np.array(all_probabilities)

# ============================================
# CONFUSION MATRIX
# ============================================

def plot_confusion_matrix(y_true, y_pred, class_names, save_path, title='Confusion Matrix'):
    # Visualize misclassifications using a heatmap
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm_percent, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Percentage (%)'}, ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")

# ============================================
# CLASSIFICATION METRICS
# ============================================

def generate_classification_report(y_true, y_pred, class_names, save_dir):
    # Calculate standard metrics like precision, recall, and F1 to see how we did
    overall_acc = accuracy_score(y_true, y_pred)
    
    # Get the detailed breakdown for each individual class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Also grab the averages to get a high-level view
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Organize everything into a readable dataframe
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    # Append the averages at the bottom for quick reference
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame({
            'Class': ['Macro Avg', 'Weighted Avg'],
            'Precision': [macro_precision, weighted_precision],
            'Recall': [macro_recall, weighted_recall],
            'F1-Score': [macro_f1, weighted_f1],
            'Support': [support.sum(), support.sum()]
        })
    ], ignore_index=True)
    
    # Dump to CSV and text file so we don't lose the results
    metrics_df.to_csv(os.path.join(save_dir, 'per_class_metrics.csv'), index=False)
    print(f"Per-class metrics saved")
    
    text_report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Overall Accuracy: {overall_acc:.4f}\n\n{text_report}")
    print(f"Classification report saved")
    
    return {
        'overall_accuracy': overall_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_metrics': metrics_df.to_dict('records')
    }

# ============================================
# PER-CLASS VISUALIZATION
# ============================================

def plot_per_class_metrics(metrics_df, save_path, title='Per-Class Performance'):
    
    # visual bar chart for per-class performance
    plot_df = metrics_df[~metrics_df['Class'].str.contains('Avg', na=False)]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(plot_df))
    width = 0.25
    
    ax.bar(x - width, plot_df['Precision'], width, label='Precision', color='steelblue')
    ax.bar(x, plot_df['Recall'], width, label='Recall', color='coral')
    ax.bar(x + width, plot_df['F1-Score'], width, label='F1-Score', color='seagreen')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['Class'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Annotate sample counts so we know if a class was underrepresented
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(i, 1.05, f"n={int(row['Support'])}", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class metrics plot saved")

# ============================================
# MALIGNANT VS BENIGN ANALYSIS
# ============================================

def analyze_malignant_vs_benign(y_true, y_pred, class_names, malignant_classes, save_dir):
    
    # Check if we missed any cancer cases, which is the most critical error
    
    # Map multi-class labels to binary (0=Benign, 1=Malignant)
    y_true_binary = np.array([1 if class_names[label] in malignant_classes else 0 
                               for label in y_true])
    y_pred_binary = np.array([1 if class_names[label] in malignant_classes else 0 
                               for label in y_pred])
    
    # Run standard stats on the binary mapping
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average=None, zero_division=0
    )
    
    # Break down the errors: False Negatives are dangerous here
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Save the binary stats to JSON
    analysis_dict = {
        'accuracy': float(accuracy),
        'benign': {
            'precision': float(precision[0]),
            'recall': float(recall[0]),
            'f1': float(f1[0]),
            'support': int(support[0])
        },
        'malignant': {
            'precision': float(precision[1]),
            'recall': float(recall[1]),
            'f1': float(f1[1]),
            'support': int(support[1])
        },
        'confusion_matrix': {
            'true_negative': int(tn),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_positive': int(tp)
        },
        'specificity': float(specificity),
        'sensitivity': float(sensitivity)
    }
    
    with open(os.path.join(save_dir, 'malignant_vs_benign_analysis.json'), 'w') as f:
        json.dump(analysis_dict, f, indent=4)
    print(f"Malignant vs Benign analysis saved")
    
    # Print a quick summary to console
    print(f"\n{'='*60}")
    print("MALIGNANT vs BENIGN ANALYSIS")
    print('='*60)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"\nBenign (n={support[0]}):")
    print(f"  Precision: {precision[0]:.4f} | Recall: {recall[0]:.4f} | F1: {f1[0]:.4f}")
    print(f"\nMalignant (n={support[1]}):")
    print(f"  Precision: {precision[1]:.4f} | Recall (Sensitivity): {recall[1]:.4f} | F1: {f1[1]:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")
    print('='*60 + "\n")
    
    return analysis_dict

# ============================================
# COMPLETE EVALUATION PIPELINE
# ============================================

def evaluate_model_complete(model_name, fold_num, test_df, device, batch_size=32):
    # Master function to load the best model and run all the tests
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name.upper()} - FOLD {fold_num}")
    print('='*80 + "\n")
    
    results_dir = os.path.join(
        Config.RESULTS_DIR, model_name, f'fold_{fold_num}', 'test_results'
    )
    os.makedirs(results_dir, exist_ok=True)
    
    # Reload the checkpoint with the best validation accuracy
    print("1. Loading best model...")
    model, checkpoint = load_best_model(model_name, fold_num, device)
    print(f"   Loaded from epoch {checkpoint['epoch']} "
          f"(Val Acc: {checkpoint['val_acc']:.2f}%)\n")
    
    # Prepare the test set (no shuffling needed)
    print("2. Creating test dataloader...")
    test_dataset = SkinLesionDataset(test_df, transform=get_val_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"   Test samples: {len(test_df)}\n")
    
    # Get predictions
    print("3. Evaluating on test set...")
    test_acc, y_pred, y_true, y_prob = evaluate_on_test_set(model, test_loader, device)
    print(f"   Test Accuracy: {test_acc:.2f}%\n")
    
    # Generate the heatmap
    print("4. Generating confusion matrix...")
    plot_confusion_matrix(
        y_true, y_pred, Config.CLASS_NAMES,
        os.path.join(results_dir, 'confusion_matrix.png'),
        f'{model_name.upper()} - Confusion Matrix'
    )
    
    # Generate the metrics report
    print("\n5. Generating classification report...")
    metrics_dict = generate_classification_report(y_true, y_pred, Config.CLASS_NAMES, results_dir)
    
    # Plot the bar charts
    print("\n6. Plotting per-class metrics...")
    metrics_df = pd.read_csv(os.path.join(results_dir, 'per_class_metrics.csv'))
    plot_per_class_metrics(
        metrics_df, os.path.join(results_dir, 'per_class_metrics.png'),
        f'{model_name.upper()} - Per-Class Performance'
    )
    
    # Run the clinical safety check
    print("\n7. Analyzing Malignant vs Benign performance...")
    malignant_analysis = analyze_malignant_vs_benign(
        y_true, y_pred, Config.CLASS_NAMES, Config.MALIGNANT_CLASSES, results_dir
    )
    
    # Bundle it all up
    evaluation_results = {
        'model_name': model_name,
        'fold_num': int(fold_num),
        'test_accuracy': float(test_acc),
        'best_epoch': int(checkpoint['epoch']),
        'val_accuracy_at_best': float(checkpoint['val_acc']),
        'metrics': metrics_dict,
        'malignant_vs_benign': malignant_analysis
    }
    
    # Save the final JSON
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=4)
    
    print(f"\n{'='*80}")
    print(f"EVALUATION COMPLETE: Test Acc = {test_acc:.2f}%")
    print(f"Results saved to: {results_dir}")
    print('='*80 + "\n")
    
    # Clear memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    return evaluation_results

# ============================================
# BEST FOLD SELECTION
# ============================================

def select_best_fold(model_name):
    # Pick the best performing fold from the JSON summary
    cv_summary_path = os.path.join(Config.RESULTS_DIR, model_name, 'cv_summary.json')
    
    if not os.path.exists(cv_summary_path):
        raise FileNotFoundError(f"CV summary not found: {cv_summary_path}")
    
    with open(cv_summary_path, 'r') as f:
        cv_summary = json.load(f)
    
    best_fold = max(cv_summary['fold_results'], key=lambda x: x['best_val_acc'])
    
    print(f"Best fold for {model_name}: Fold {best_fold['fold']} "
          f"(Val Acc: {best_fold['best_val_acc']:.2f}%)")
    
    return best_fold['fold'], best_fold['best_val_acc']

# ============================================
# MODEL COMPARISON
# ============================================

def compare_models_on_test_set(model_names, test_df, device):
    # Run all models against the test set and log the results
    print("\n" + "="*80)
    print("COMPARING MODELS ON TEST SET")
    print("="*80 + "\n")
    
    comparison_results = []
    
    for model_name in model_names:
        print(f"\nProcessing {model_name.upper()}...")
        print("-"*80)
        
        best_fold, _ = select_best_fold(model_name)
        results = evaluate_model_complete(model_name, best_fold, test_df, device, batch_size=32)
        comparison_results.append(results)
    
    # Consolidate metrics into a clean dataframe
    comparison_data = []
    for result in comparison_results:
        comparison_data.append({
            'Model': result['model_name'],
            'Best Fold': result['fold_num'],
            'Val Acc (%)': result['val_accuracy_at_best'],
            'Test Acc (%)': result['test_accuracy'],
            'Macro F1': result['metrics']['macro_f1'],
            'Weighted F1': result['metrics']['weighted_f1'],
            'Malignant Sensitivity': result['malignant_vs_benign']['sensitivity'],
            'Benign Specificity': result['malignant_vs_benign']['specificity']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save the comparison data for later
    comparison_dir = os.path.join(Config.RESULTS_DIR, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    csv_path = os.path.join(comparison_dir, 'model_comparison_test_results.csv')
    comparison_df.to_csv(csv_path, index=False)
    
    # Dump the table to the console
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")
    print(f"Comparison saved to: {csv_path}")
    
    return comparison_results, comparison_df

# ============================================
# MODEL COMPARISON VISUALIZATIONS
# ============================================

def plot_model_comparison(comparison_df, save_dir):
    
    # Visualize how models stack up against each other
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    models = comparison_df['Model']
    x = np.arange(len(models))
    width = 0.35
    
    # Basic accuracy comparison
    axes[0, 0].bar(x, comparison_df['Test Acc (%)'], color='steelblue')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Test Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(comparison_df['Test Acc (%)']):
        axes[0, 0].text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')
    
    # F1 score breakdown
    axes[0, 1].bar(x - width/2, comparison_df['Macro F1'], width, 
                   label='Macro F1', color='coral')
    axes[0, 1].bar(x + width/2, comparison_df['Weighted F1'], width, 
                   label='Weighted F1', color='seagreen')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('F1 Score Comparison', fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Clinical metrics: missing cancer vs false alarms
    axes[1, 0].bar(x - width/2, comparison_df['Malignant Sensitivity'], width, 
                   label='Malignant Sensitivity', color='crimson')
    axes[1, 0].bar(x + width/2, comparison_df['Benign Specificity'], width, 
                   label='Benign Specificity', color='dodgerblue')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Clinical Performance (Malignant vs Benign)', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].set_ylim([0, 1.1])
    
    # Check for overfitting (val vs test)
    axes[1, 1].scatter(comparison_df['Val Acc (%)'], comparison_df['Test Acc (%)'], 
                       s=200, alpha=0.6, c=['steelblue', 'coral', 'seagreen'])
    for i, model in enumerate(models):
        axes[1, 1].annotate(model, 
                           (comparison_df.iloc[i]['Val Acc (%)'], 
                            comparison_df.iloc[i]['Test Acc (%)']),
                           xytext=(5, 5), textcoords='offset points')
    
    min_val = min(comparison_df['Val Acc (%)'].min(), comparison_df['Test Acc (%)'].min())
    max_val = max(comparison_df['Val Acc (%)'].max(), comparison_df['Test Acc (%)'].max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, 
                    label='Perfect Generalization')
    axes[1, 1].set_xlabel('Validation Accuracy (%)')
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_title('Generalization Analysis', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Comparison on Test Set', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison_charts.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison charts saved")

def create_confusion_matrix_comparison(model_names, test_df, save_dir):
    # Generate side-by-side confusion matrices for easy comparison
    device = get_device()
    
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(10*n_models, 8))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, model_name in enumerate(model_names):
        best_fold, _ = select_best_fold(model_name)
        
        model, _ = load_best_model(model_name, best_fold, device)
        test_dataset = SkinLesionDataset(test_df, transform=get_val_test_transforms())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        _, y_pred, y_true, _ = evaluate_on_test_set(model, test_loader, device)
        
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=Config.CLASS_NAMES, yticklabels=Config.CLASS_NAMES,
                    cbar_kws={'label': 'Percentage (%)'}, ax=axes[idx])
        
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label' if idx == 0 else '')
        axes[idx].set_title(f'{model_name.upper()}', fontweight='bold')
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    plt.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrices_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices comparison saved")

# ============================================
# GRAD-CAM IMPLEMENTATION
# ============================================

class GradCAM:
    
    # Helper class to visualize where the model is looking
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook into the model to capture gradients for the heatmap
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Run a backward pass to get gradients for the specific class
        self.model.eval()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        # Average the gradients (GAP)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Weight the activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU to keep only positive contributions and normalize
        cam = torch.clamp(cam, min=0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()

def apply_colormap_on_image(org_img, activation_map, colormap=cv2.COLORMAP_JET, alpha=0.5):
    # Overlay the heatmap onto the original image
    activation_map_resized = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * activation_map_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255.0
    
    overlayed_img = alpha * heatmap + (1 - alpha) * org_img
    return np.clip(overlayed_img, 0, 1)

def visualize_gradcam_samples(model_name, fold_num, test_df, device, num_samples=10, save_dir=None):
    # Skip ViT since it's not a CNN, then generate heatmaps for random samples

    # Transformers don't work with standard Grad-CAM, so skip
    if model_name == 'vit_b16':
        print(f"\n{'='*80}")
        print(f"GRAD-CAM VISUALIZATION: {model_name.upper()}")
        print('='*80)
        print("  Grad-CAM is designed for CNN architectures.")
        print("  Vision Transformers use attention mechanisms instead of")
        print("  convolutional feature maps. Skipping Grad-CAM for ViT-B/16.")
        print('='*80 + "\n")
        return

    print(f"\n{'='*80}")
    print(f"GENERATING GRAD-CAM VISUALIZATIONS: {model_name.upper()}")
    print('='*80 + "\n")
    
    if save_dir is None:
        save_dir = os.path.join(
            Config.RESULTS_DIR, model_name, f'fold_{fold_num}',
            'test_results', 'gradcam_samples'
        )
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model and setup hooks
    model, _ = load_best_model(model_name, fold_num, device)
    target_layer = get_model_grad_cam_layer(model_name, model)
    gradcam = GradCAM(model, target_layer)
    
    # Get all predictions first to pick interesting samples
    test_dataset = SkinLesionDataset(test_df, transform=get_val_test_transforms())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    _, all_preds, all_labels, _ = evaluate_on_test_set(model, test_loader, device)
    
    # Split into rights and wrongs
    correct_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p == l]
    incorrect_indices = [i for i, (p, l) in enumerate(zip(all_preds, all_labels)) if p != l]
    
    num_correct = min(num_samples // 2, len(correct_indices))
    num_incorrect = min(num_samples // 2, len(incorrect_indices))
    
    # Randomly select a mix of both
    np.random.seed(Config.RANDOM_SEED)
    selected_correct = np.random.choice(correct_indices, num_correct, replace=False) if correct_indices else []
    selected_incorrect = np.random.choice(incorrect_indices, num_incorrect, replace=False) if incorrect_indices else []
    selected_indices = list(selected_correct) + list(selected_incorrect)
    
    print(f"Generating Grad-CAM for {len(selected_indices)} samples...")
    print(f"  Correct: {num_correct} | Incorrect: {num_incorrect}\n")
    
    # Generate and save the visualization for each selected sample
    for idx, sample_idx in enumerate(tqdm(selected_indices, desc='Generating Grad-CAM')):
        img_path = test_df.iloc[sample_idx]['image_path']
        true_label = test_df.iloc[sample_idx]['label']
        pred_label = all_preds[sample_idx]
        
        # Load image manually
        original_img = np.array(Image.open(img_path).convert('RGB')) / 255.0
        img_tensor, _ = test_dataset[sample_idx]
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get the heatmap
        cam = gradcam.generate_cam(img_tensor, target_class=pred_label)
        
        # Combine them
        original_img_resized = cv2.resize(original_img, (Config.IMG_SIZE, Config.IMG_SIZE))
        overlayed_img = apply_colormap_on_image(original_img_resized, cam, alpha=0.5)
        
        # Plot side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_img_resized)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap', fontsize=12)
        axes[1].axis('off')
        
        axes[2].imshow(overlayed_img)
        axes[2].set_title('Overlay', fontsize=12)
        axes[2].axis('off')
        
        is_correct = (true_label == pred_label)
        status = "Correct" if is_correct else "Incorrect"
        fig.suptitle(
            f"{status} | True: {Config.CLASS_NAMES[true_label]} | "
            f"Pred: {Config.CLASS_NAMES[pred_label]}",
            fontsize=14, fontweight='bold',
            color='green' if is_correct else 'red'
        )
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f'gradcam_{idx+1}_{"correct" if is_correct else "incorrect"}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
    
    print(f"\nGrad-CAM visualizations saved to: {save_dir}\n")
    
    del model, gradcam
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

def create_gradcam_comparison_grid(model_names, test_df, device, num_samples=3):
    
    # Build a side-by-side heatmap comparison for CNN models
    print(f"\n{'='*80}")
    print("CREATING GRAD-CAM COMPARISON GRID")
    print('='*80 + "\n")

    # Remove ViT from this list as it lacks standard feature maps
    model_names_filtered = [m for m in model_names if m != 'vit_b16']
    
    if not model_names_filtered:
        print("No models available for Grad-CAM comparison")
        return
    
    if 'vit_b16' in model_names:
        print("Note: ViT-B/16 excluded (Transformer architecture)")
        print("Comparing CNN models only: ResNet18 & EfficientNet-B0\n")
    
    comparison_dir = os.path.join(Config.RESULTS_DIR, 'comparison', 'gradcam_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    np.random.seed(Config.RANDOM_SEED)
    sample_indices = np.random.choice(len(test_df), num_samples, replace=False)
    
    # Loop through selected samples and generate plots for each model
    for sample_idx in sample_indices:
        img_path = test_df.iloc[sample_idx]['image_path']
        true_label = test_df.iloc[sample_idx]['label']
        
        original_img = np.array(Image.open(img_path).convert('RGB')) / 255.0
        original_img_resized = cv2.resize(original_img, (Config.IMG_SIZE, Config.IMG_SIZE))
        
        n_models = len(model_names)
        fig, axes = plt.subplots(n_models, 3, figsize=(12, 4*n_models))
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for model_idx, model_name in enumerate(model_names_filtered):
            best_fold, _ = select_best_fold(model_name)
            model, _ = load_best_model(model_name, best_fold, device)
            
            test_dataset = SkinLesionDataset(test_df, transform=get_val_test_transforms())
            img_tensor, _ = test_dataset[sample_idx]
            img_tensor_batch = img_tensor.unsqueeze(0).to(device)
            
            model.eval()
            with torch.no_grad():
                pred_label = model(img_tensor_batch).argmax(dim=1).item()
            
            # Generate the heatmap using our hook-based class
            target_layer = get_model_grad_cam_layer(model_name, model)
            gradcam = GradCAM(model, target_layer)
            cam = gradcam.generate_cam(img_tensor_batch, target_class=pred_label)
            overlayed_img = apply_colormap_on_image(original_img_resized, cam, alpha=0.5)
            
            axes[model_idx, 0].imshow(original_img_resized)
            axes[model_idx, 0].set_title(f'{model_name.upper()}\nOriginal')
            axes[model_idx, 0].axis('off')
            
            axes[model_idx, 1].imshow(cam, cmap='jet')
            axes[model_idx, 1].set_title('Heatmap')
            axes[model_idx, 1].axis('off')
            
            is_correct = (true_label == pred_label)
            axes[model_idx, 2].imshow(overlayed_img)
            axes[model_idx, 2].set_title(
                f'{"OK " if is_correct else "NO "} Pred: {Config.CLASS_NAMES[pred_label]}'
            )
            axes[model_idx, 2].axis('off')
            
            del model, gradcam
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        plt.suptitle(
            f'Grad-CAM Comparison\nTrue Label: {Config.CLASS_NAMES[true_label]}',
            fontsize=14, fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, f'comparison_sample_{sample_idx}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()
    
    print(f"Grad-CAM comparison grid saved to: {comparison_dir}\n")

# ============================================
# MAIN EXECUTION PIPELINE
# ============================================

def initialize_data_split():
    # Setup dataframes, loading from disk if they exist to keep consistency
    print("\n" + "="*80)
    print("INITIALIZING DATA")
    print("="*80 + "\n")
    
    split_dir = os.path.join(Config.RESULTS_DIR, 'data_split')
    
    # Check for cached split
    train_df, test_df = load_data_split(split_dir)
    
    if train_df is None:
        print("Creating new data split...")
        
        if not os.path.exists(Config.DATA_DIR):
            raise FileNotFoundError(f"Data directory not found: {Config.DATA_DIR}")
        
        # Scrape and split the raw data
        df = collect_image_paths(Config.DATA_DIR)
        print(f"Found {len(df)} images across {df['label'].nunique()} classes")
        
        train_df, test_df = create_train_test_split(
            df, 
            test_size=1-Config.TRAIN_TEST_SPLIT,
            random_state=Config.RANDOM_SEED
        )
        
        # Persist splits for future runs
        save_data_split(train_df, test_df, split_dir)
        plot_class_distribution(train_df, test_df, split_dir)
    
    print(f"\nData ready: {len(train_df)} train | {len(test_df)} test")
    print("="*80 + "\n")
    
    return train_df, test_df

def main_full_training():
    
    # Orchestrator for the entire training phase
    print("STARTING FULL TRAINING PIPELINE")
    
    device = get_device()
    set_seed(Config.RANDOM_SEED)
    
    train_df, test_df = initialize_data_split()
    
    model_names = ['resnet18', 'efficientnet_b0', 'vit_b16']
    
    print(f"Training Configuration:")
    print(f"  Models: {', '.join(model_names)}")
    print(f"  Cross-Validation: {Config.N_FOLDS} folds")
    print(f"  Max Epochs: {Config.NUM_EPOCHS}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Device: {device}\n")
    
    # Train each model using K-Fold CV
    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"TRAINING {model_name.upper()}")
        print('='*80 + "\n")
        
        cv_results = train_with_cross_validation(
            model_name=model_name,
            train_df=train_df,
            n_folds=Config.N_FOLDS,
            num_epochs=Config.NUM_EPOCHS,
            batch_size=Config.BATCH_SIZE,
            device=device
        )
        
        print(f"\n {model_name.upper()} training completed!")
        print(f"  Mean CV Accuracy: {cv_results['mean_val_acc']:.2f} ± "
              f"{cv_results['std_val_acc']:.2f}%\n")

    print("ALL MODELS TRAINED SUCCESSFULLY!")

def main_full_evaluation():
    
    # Orchestrator for testing, comparison, and visualization
    print("\n" + "=="*20)
    print("STARTING FULL EVALUATION PIPELINE")
    print("=="*20 + "\n")
    
    device = get_device()
    set_seed(Config.RANDOM_SEED)
    
    _, test_df = initialize_data_split()
    
    model_names = ['resnet18', 'efficientnet_b0', 'vit_b16']
    
    # Step 1: Run standard metrics for each model
    print("="*80)
    print("STEP 1: EVALUATING INDIVIDUAL MODELS")
    print("="*80 + "\n")
    
    for model_name in model_names:
        best_fold, _ = select_best_fold(model_name)
        evaluate_model_complete(model_name, best_fold, test_df, device)
    
    # Step 2: Compare performance across models
    print("\n" + "="*80)
    print("STEP 2: COMPARING MODELS")
    print("="*80 + "\n")
    
    comparison_results, comparison_df = compare_models_on_test_set(
        model_names, test_df, device
    )
    
    # Step 3: Generate comparative charts
    print("\n" + "="*80)
    print("STEP 3: CREATING COMPARISON VISUALIZATIONS")
    print("="*80 + "\n")
    
    comparison_dir = os.path.join(Config.RESULTS_DIR, 'comparison')
    
    print("Creating comparison charts...")
    plot_model_comparison(comparison_df, comparison_dir)
    
    print("Creating confusion matrix comparison...")
    create_confusion_matrix_comparison(model_names, test_df, comparison_dir)
    
    # Step 4: Visual interpretability (Grad-CAM)
    print("\n" + "="*80)
    print("STEP 4: GENERATING GRAD-CAM VISUALIZATIONS")
    print("="*80 + "\n")
    
    for model_name in model_names:
        best_fold, _ = select_best_fold(model_name)
        visualize_gradcam_samples(
            model_name, best_fold, test_df, device, num_samples=10
        )
    
    print("Creating Grad-CAM comparison grid...")
    create_gradcam_comparison_grid(model_names, test_df, device, num_samples=3)
    
    # Step 5: Wrap up
    print("\n" + "=="*20)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=="*20 + "\n")
    
    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")
    
    print(f"All results saved to: {Config.RESULTS_DIR}")

# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    import argparse
    
    # setup command line argument parsing
    parser = argparse.ArgumentParser(
        description='Multi-Class Skin Lesion Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model.py --mode train      # Train all models
  python model.py --mode evaluate   # Evaluate trained models
  python model.py --mode full       # Complete pipeline (train + evaluate)
        """
    )
    
    # define execution mode argument
    parser.add_argument(
        '--mode', 
        type=str, 
        default='full',
        choices=['train', 'evaluate', 'full'],
        help='Execution mode'
    )
    
    args = parser.parse_args()
    
    # print header info
    print("\n" + "="*80)
    print("EEE 443/543 - SKIN LESION CLASSIFICATION PROJECT")
    print("Author: Sefa Emre Kavgacı (22103179)")
    print("="*80)
    
    try:
        # execute training pipeline
        if args.mode == 'train':
            main_full_training()
        
        # execute evaluation pipeline
        elif args.mode == 'evaluate':
            main_full_evaluation()
        
        # run both training and evaluation sequentially
        elif args.mode == 'full':
            print("RUNNING COMPLETE PIPELINE: TRAINING + EVALUATION")
            
            main_full_training()
            main_full_evaluation()
            
            print("COMPLETE PIPELINE FINISHED!")
    
    # handle manual interruption
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    
    # catch and log unexpected errors
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        raise