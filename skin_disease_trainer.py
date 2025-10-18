#!/usr/bin/env python3
"""
Comprehensive Skin Disease Classification Training Pipeline
Implements multiple CNN architectures with transfer learning, hyperparameter tuning,
early stopping, and comprehensive evaluation metrics.
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import efficientnet_b0, mobilenet_v3_large

from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

class SkinDiseaseTrainer:
    """Comprehensive trainer for skin disease classification models"""
    
    def __init__(self, data_dir: str = 'final_dataset', output_dir: str = 'training_results'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training configuration
        self.config = {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 15,
            'min_delta': 0.001,
            'num_workers': 4,
            'pin_memory': True if self.device.type == 'cuda' else False
        }
        
        # Model architectures to train
        self.architectures = {
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'mobilenet_v3': mobilenet_v3_large,
            'densenet121': models.densenet121,
            'efficientnet_b0': efficientnet_b0
        }
        
        self.results = {}
        self.class_names = None
        self.num_classes = None
        
    def setup_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Setup data loaders with augmentation and class balancing"""
        print("Setting up data loaders...")
        
        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(self.data_dir / 'train', transform=train_transform)
        val_dataset = datasets.ImageFolder(self.data_dir / 'val', transform=val_test_transform)
        test_dataset = datasets.ImageFolder(self.data_dir / 'test', transform=val_test_transform)
        
        self.class_names = train_dataset.classes
        self.num_classes = len(self.class_names)
        
        print(f"Found {self.num_classes} classes: {self.class_names}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Calculate class weights for imbalanced dataset
        class_counts = [0] * self.num_classes
        for _, label in train_dataset:
            class_counts[label] += 1
        
        total_samples = sum(class_counts)
        class_weights = [total_samples / (self.num_classes * count) for count in class_counts]
        class_weights_tensor = torch.FloatTensor(class_weights)
        
        # Create weighted sampler
        sample_weights = [class_weights[label] for _, label in train_dataset]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=self.config['pin_memory']
        )
        
        return train_loader, val_loader, test_loader, class_weights_tensor
    
    def create_model(self, architecture: str) -> nn.Module:
        """Create model with transfer learning"""
        print(f"Creating {architecture} model...")
        
        if architecture not in self.architectures:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Load pretrained model
        model_func = self.architectures[architecture]
        
        if architecture == 'efficientnet_b0':
            model = model_func(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
        elif architecture == 'mobilenet_v3':
            model = model_func(pretrained=True)
            model.classifier = nn.Linear(model.classifier[-1].in_features, self.num_classes)
        elif 'resnet' in architecture:
            model = model_func(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        elif 'densenet' in architecture:
            model = model_func(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, self.num_classes)
        
        return model.to(self.device)
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, class_weights: torch.Tensor,
                   architecture: str) -> Dict:
        """Train a single model with comprehensive monitoring"""
        print(f"\n{'='*60}")
        print(f"Training {architecture.upper()}")
        print(f"{'='*60}")
        
        # Setup training components
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rates': []
        }
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                # Print progress every 100 batches
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch+1}/{self.config["epochs"]}, '
                          f'Batch {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}, '
                          f'Acc: {100.*train_correct/train_total:.2f}%')
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['learning_rates'].append(current_lr)
            
            print(f'Epoch {epoch+1}/{self.config["epochs"]}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  Learning Rate: {current_lr:.6f}')
            
            # Early stopping and model saving
            if val_acc > best_val_acc + self.config['min_delta']:
                best_val_acc = val_acc
                best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                model_path = self.output_dir / 'models' / f'{architecture}_best.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_names': self.class_names
                }, model_path)
                print(f'  New best model saved! Val Acc: {val_acc:.2f}%')
            else:
                patience_counter += 1
                print(f'  No improvement ({patience_counter}/{self.config["patience"]})')
            
            if patience_counter >= self.config['patience']:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            print('-' * 50)
        
        training_time = time.time() - start_time
        print(f'Training completed in {training_time:.2f} seconds')
        print(f'Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch+1}')
        
        return {
            'history': history,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'training_time': training_time,
            'architecture': architecture
        }
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                      architecture: str) -> Dict:
        """Comprehensive model evaluation"""
        print(f"\nEvaluating {architecture}...")
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = 100. * np.mean(np.array(all_predictions) == np.array(all_targets))
        
        # Classification report
        report = classification_report(
            all_targets, all_predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # ROC curves for multiclass
        y_test_bin = label_binarize(all_targets, classes=range(self.num_classes))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], 
                                         np.array(all_probabilities)[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Average ROC AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), 
                                                 np.array(all_probabilities).ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def plot_training_history(self, history: Dict, architecture: str):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{architecture.upper()} Training History', fontsize=16)
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(history['val_acc'], label='Validation Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning Rate
        axes[1, 0].plot(history['learning_rates'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        
        # Combined plot
        ax2 = axes[1, 0].twinx()
        ax2.plot(history['val_acc'], color='red', alpha=0.7)
        ax2.set_ylabel('Validation Accuracy (%)', color='red')
        
        # Remove empty subplot
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / f'{architecture}_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, architecture: str):
        """Plot confusion matrix"""
        plt.figure(figsize=(15, 12))
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        
        plt.title(f'{architecture.upper()} - Confusion Matrix (Normalized)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / f'{architecture}_confusion_matrix.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, roc_auc: Dict, architecture: str):
        """Plot ROC curves"""
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curves for each class
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        for i, color in zip(range(self.num_classes), colors):
            plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            if i in roc_auc:
                plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, 
                        label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, 
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{architecture.upper()} - ROC Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / f'{architecture}_roc_curves.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_full_training(self):
        """Run training for all architectures"""
        print("=" * 80)
        print("SKIN DISEASE CLASSIFICATION - COMPREHENSIVE MODEL TRAINING")
        print("=" * 80)
        
        # Setup data
        train_loader, val_loader, test_loader, class_weights = self.setup_data_loaders()
        
        # Train each architecture
        for architecture in self.architectures.keys():
            try:
                # Create model
                model = self.create_model(architecture)
                
                # Train model
                training_result = self.train_model(
                    model, train_loader, val_loader, class_weights, architecture
                )
                
                # Load best model for evaluation
                model_path = self.output_dir / 'models' / f'{architecture}_best.pth'
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Evaluate model
                evaluation_result = self.evaluate_model(model, test_loader, architecture)
                
                # Store results
                self.results[architecture] = {
                    'training': training_result,
                    'evaluation': evaluation_result
                }
                
                # Generate plots
                self.plot_training_history(training_result['history'], architecture)
                self.plot_confusion_matrix(evaluation_result['confusion_matrix'], architecture)
                self.plot_roc_curves(evaluation_result['roc_auc'], architecture)
                
                # Save detailed report
                self.save_model_report(architecture, training_result, evaluation_result)
                
                print(f"\n{architecture.upper()} completed successfully!")
                print(f"Test Accuracy: {evaluation_result['accuracy']:.2f}%")
                
            except Exception as e:
                print(f"Error training {architecture}: {str(e)}")
                continue
        
        # Generate comparison report
        self.generate_comparison_report()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED!")
        print("=" * 80)
    
    def save_model_report(self, architecture: str, training_result: Dict, 
                         evaluation_result: Dict):
        """Save detailed model report"""
        report = {
            'architecture': architecture,
            'training_time': training_result['training_time'],
            'best_epoch': training_result['best_epoch'],
            'best_val_acc': training_result['best_val_acc'],
            'test_accuracy': evaluation_result['accuracy'],
            'classification_report': evaluation_result['classification_report'],
            'roc_auc_scores': evaluation_result['roc_auc']
        }
        
        with open(self.output_dir / 'reports' / f'{architecture}_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("No results to compare!")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for arch, result in self.results.items():
            comparison_data.append({
                'Architecture': arch.upper(),
                'Training Time (min)': result['training']['training_time'] / 60,
                'Best Epoch': result['training']['best_epoch'],
                'Best Val Acc (%)': result['training']['best_val_acc'],
                'Test Acc (%)': result['evaluation']['accuracy'],
                'Avg ROC AUC': np.mean(list(result['evaluation']['roc_auc'].values()))
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test Acc (%)', ascending=False)
        
        # Save comparison
        df.to_csv(self.output_dir / 'reports' / 'model_comparison.csv', index=False)
        
        # Print comparison
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))
        
        # Find best model
        best_model = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_model['Architecture']}")
        print(f"   Test Accuracy: {best_model['Test Acc (%)']:.2f}%")
        print(f"   Training Time: {best_model['Training Time (min)']:.1f} minutes")
        
        # Save best model info
        with open(self.output_dir / 'best_model.json', 'w') as f:
            json.dump(best_model.to_dict(), f, indent=2)

def main():
    """Main training function"""
    trainer = SkinDiseaseTrainer()
    trainer.run_full_training()

if __name__ == "__main__":
    main()
