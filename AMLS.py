#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from scipy import signal
from scipy.fft import fft, ifft
import os
import warnings
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Any, Union
import torchaudio.transforms as T
import struct
import zipfile

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

def plot_training_history(history: Dict[str, Union[List[float], float]],
                         model_name: str = "Model",
                         save_path: Optional[str] = None,
                         figsize: tuple = (20, 12),
                         style: str = 'whitegrid'):

    def ensure_list(value):
        """Convert single values to lists for consistent handling"""
        if isinstance(value, (int, float)):
            return [value]
        return value

    def is_valid_series(key):
        """Check if a key exists and contains valid data for plotting"""
        return key in history and history[key] is not None and (
            isinstance(history[key], list) and len(history[key]) > 0 or
            isinstance(history[key], (int, float))
        )

    sns.set_style(style)
    plt.style.use('seaborn-v0_8')

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold')

    train_color = '#2E86AB'
    val_color = '#A23B72'
    lr_color = '#F18F01'

    ax1 = axes[0, 0]
    if is_valid_series('train_losses') and is_valid_series('val_losses'):
        train_losses = ensure_list(history['train_losses'])
        val_losses = ensure_list(history['val_losses'])

        max_len = max(len(train_losses), len(val_losses))
        epochs = range(1, max_len + 1)

        if len(train_losses) == max_len:
            ax1.plot(epochs, train_losses, color=train_color, label='Training Loss', linewidth=2)
        if len(val_losses) == max_len:
            ax1.plot(epochs, val_losses, color=val_color, label='Validation Loss', linewidth=2)

        ax1.set_title('Training & Validation Loss', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        if len(val_losses) > 1:
            min_val_loss = min(val_losses)
            min_epoch = val_losses.index(min_val_loss) + 1
            ax1.annotate(f'Min Val Loss: {min_val_loss:.4f}\nEpoch: {min_epoch}',
                        xy=(min_epoch, min_val_loss), xytext=(min_epoch + len(epochs)*0.1, min_val_loss),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        else:
            ax1.text(0.5, 0.5, f'Final Val Loss: {val_losses[0]:.4f}',
                    ha='center', va='center', transform=ax1.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    else:
        ax1.text(0.5, 0.5, 'Loss data not available',
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Training & Validation Loss', fontweight='bold')

    ax2 = axes[0, 1]
    if is_valid_series('train_accs') and is_valid_series('val_accs'):
        train_accs = ensure_list(history['train_accs'])
        val_accs = ensure_list(history['val_accs'])

        max_len = max(len(train_accs), len(val_accs))
        epochs = range(1, max_len + 1)

        if len(train_accs) == max_len:
            ax2.plot(epochs, train_accs, color=train_color, label='Training Accuracy', linewidth=2)
        if len(val_accs) == max_len:
            ax2.plot(epochs, val_accs, color=val_color, label='Validation Accuracy', linewidth=2)

        ax2.set_title('Training & Validation Accuracy', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        if len(val_accs) > 1:
            max_val_acc = max(val_accs)
            max_epoch = val_accs.index(max_val_acc) + 1
            ax2.annotate(f'Max Val Acc: {max_val_acc:.4f}\nEpoch: {max_epoch}',
                        xy=(max_epoch, max_val_acc), xytext=(max_epoch + len(epochs)*0.1, max_val_acc),
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        else:
            ax2.text(0.5, 0.5, f'Final Val Acc: {val_accs[0]:.4f}',
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    else:
        ax2.text(0.5, 0.5, 'Accuracy data not available',
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Training & Validation Accuracy', fontweight='bold')

    ax3 = axes[0, 2]
    if is_valid_series('best_f1'):
        best_f1 = ensure_list(history['best_f1'])
        epochs = range(1, len(best_f1) + 1)
        ax3.plot(epochs, best_f1, color=val_color, label='Best F1 Score', linewidth=2)
        ax3.set_title('Best F1 Score', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        if len(best_f1) > 1:
            max_f1 = max(best_f1)
            max_epoch = best_f1.index(max_f1) + 1
            ax3.annotate(f'Max F1: {max_f1:.4f}\nEpoch: {max_epoch}',
                        xy=(max_epoch, max_f1), xytext=(max_epoch + len(epochs)*0.1, max_f1),
                        arrowprops=dict(arrowstyle='->', color='purple', alpha=0.7),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        else:
            ax3.text(0.5, 0.5, f'Final F1: {best_f1[0]:.4f}',
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    else:
        ax3.text(0.5, 0.5, 'F1 Score data not available',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Best F1 Score', fontweight='bold')


    ax4 = axes[1, 0]
    if is_valid_series('train_losses') and is_valid_series('val_losses'):
        train_losses = ensure_list(history['train_losses'])
        val_losses = ensure_list(history['val_losses'])

        if len(train_losses) > 1 and len(val_losses) > 1:
            min_len = min(len(train_losses), len(val_losses))
            epochs = range(1, min_len + 1)
            loss_gap = [train_losses[i] - val_losses[i] for i in range(min_len)]

            ax4.plot(epochs, loss_gap, color='red', label='Train-Val Loss Gap', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax4.set_title('Overfitting Analysis', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Loss Gap (Train - Val)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            final_gap = loss_gap[-1]
            if final_gap > 0:
                ax4.text(0.7, 0.9, f'Final Gap: {final_gap:.4f}\n(Overfitting)',
                        transform=ax4.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
            else:
                ax4.text(0.7, 0.9, f'Final Gap: {final_gap:.4f}\n(Underfitting)',
                        transform=ax4.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for\noverfitting analysis',
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Overfitting Analysis', fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Loss data not available\nfor overfitting analysis',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Overfitting Analysis', fontweight='bold')


    ax5 = axes[1, 1]
    if (is_valid_series('train_losses') and is_valid_series('val_losses') and
        is_valid_series('train_accs') and is_valid_series('val_accs')):

        train_losses = ensure_list(history['train_losses'])
        val_losses = ensure_list(history['val_losses'])
        train_accs = ensure_list(history['train_accs'])
        val_accs = ensure_list(history['val_accs'])

        max_len = max(len(train_losses), len(val_losses), len(train_accs), len(val_accs))
        epochs = range(1, max_len + 1)

        ax5_twin = ax5.twinx()

        if len(train_losses) == max_len:
            line1 = ax5.plot(epochs, train_losses, color=train_color, label='Train Loss', linewidth=2)
        if len(val_losses) == max_len:
            line2 = ax5.plot(epochs, val_losses, color=val_color, label='Val Loss', linewidth=2)

        if len(train_accs) == max_len:
            line3 = ax5_twin.plot(epochs, train_accs, color=train_color, label='Train Acc', linewidth=2, linestyle='--')
        if len(val_accs) == max_len:
            line4 = ax5_twin.plot(epochs, val_accs, color=val_color, label='Val Acc', linewidth=2, linestyle='--')

        ax5.set_title('Loss vs Accuracy', fontweight='bold')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss', color='black')
        ax5_twin.set_ylabel('Accuracy', color='black')

        lines = []
        if len(train_losses) == max_len:
            lines.extend(line1)
        if len(val_losses) == max_len:
            lines.extend(line2)
        if len(train_accs) == max_len:
            lines.extend(line3)
        if len(val_accs) == max_len:
            lines.extend(line4)

        if lines:
            labels = [l.get_label() for l in lines]
            ax5.legend(lines, labels, loc='center right')

        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Loss and Accuracy data\nnot available',
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Loss vs Accuracy', fontweight='bold')

    ax6 = axes[1, 2]

    summary_stats = []

    if is_valid_series('val_losses'):
        val_losses = ensure_list(history['val_losses'])
        summary_stats.append(f"Best Val Loss: {min(val_losses):.4f}")

    if is_valid_series('val_accs'):
        val_accs = ensure_list(history['val_accs'])
        summary_stats.append(f"Best Val Accuracy: {max(val_accs):.4f}")

    if is_valid_series('best_f1'):
        best_f1 = ensure_list(history['best_f1'])
        summary_stats.append(f"Best F1 Score: {max(best_f1):.4f}")

    if is_valid_series('val_losses'):
        val_losses = ensure_list(history['val_losses'])
        if len(val_losses) > 10:
            recent_loss = val_losses[-10:]
            loss_std = np.std(recent_loss)
            summary_stats.append(f"Recent Loss Std: {loss_std:.6f}")

    if is_valid_series('train_losses') and is_valid_series('val_losses'):
        train_losses = ensure_list(history['train_losses'])
        val_losses = ensure_list(history['val_losses'])
        if len(train_losses) > 0 and len(val_losses) > 0:
            final_gap = train_losses[-1] - val_losses[-1]
            summary_stats.append(f"Final Train-Val Gap: {final_gap:.4f}")

    if is_valid_series('best_f1'):
        best_f1 = ensure_list(history['best_f1'])
        summary_stats.append(f"Total Epochs: {len(best_f1)}")
        if len(best_f1) > 1:
            max_f1_epoch = best_f1.index(max(best_f1)) + 1
            summary_stats.append(f"Best F1 at Epoch: {max_f1_epoch}")

    ax6.text(0.1, 0.9, "Training Summary:", transform=ax6.transAxes,
             fontsize=14, fontweight='bold', verticalalignment='top')

    for i, stat in enumerate(summary_stats):
        ax6.text(0.1, 0.8 - i*0.08, stat, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top')

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_metrics_comparison(histories: Dict[str, Dict[str, Union[List[float], float]]],
                          metric: str = 'val_f1',
                          title: str = None,
                          save_path: Optional[str] = None,
                          figsize: tuple = (12, 8)):

    def ensure_list(value):
        """Convert single values to lists for consistent handling"""
        if isinstance(value, (int, float)):
            return [value]
        return value

    plt.figure(figsize=figsize)

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590']

    for i, (model_name, history) in enumerate(histories.items()):
        if metric in history and history[metric] is not None:
            metric_values = ensure_list(history[metric])
            epochs = range(1, len(metric_values) + 1)
            plt.plot(epochs, metric_values,
                    color=colors[i % len(colors)],
                    label=model_name,
                    linewidth=2.5,
                    marker='o' if len(epochs) <= 50 else None,
                    markersize=4)

    plt.title(title or f'{metric.replace("_", " ").title()} Comparison', fontweight='bold', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

    plt.show()

def read_zip_binary(path):
    """Read binary data from zip file"""
    ragged_array = []
    with zipfile.ZipFile(path, 'r') as zf:
        inner_path = path.split("/")[-1].split(".")[0]
        print(f"Reading {inner_path}.bin from {path}")
        with zf.open(f'{inner_path}.bin', 'r') as r:
            read_binary_from(ragged_array, r)
    return ragged_array

def read_binary(path):
    """Read binary data from unzipped file"""
    ragged_array = []
    with open(path, "rb") as r:
        read_binary_from(ragged_array, r)
    return ragged_array

def read_binary_from(ragged_array, r):
    """Helper function to read binary data"""
    while True:
        size_bytes = r.read(4)
        if not size_bytes:
            break
        sub_array_size = struct.unpack('i', size_bytes)[0]
        sub_array = list(struct.unpack(f'{sub_array_size}h', r.read(sub_array_size * 2)))
        ragged_array.append(sub_array)

def load_labels(path):
    """Load labels from CSV file"""
    df = pd.read_csv(path, header=None) # Assuming no header based on the user's description of a single column
    return df[0].values

class ECGDataExplorer:
    def __init__(self, X_data, y_labels):
        self.X_data = X_data
        self.y_labels = y_labels
        self.class_names = ['Normal', 'AF', 'Other', 'Noisy']

    def analyze_data(self):
        """Perform comprehensive data analysis"""
        print("=== ECG Data Analysis ===")

        lengths = [len(x) for x in self.X_data]
        print(f"Total samples: {len(self.X_data)}")
        print(f"Length statistics:")
        print(f"  Min: {min(lengths)}, Max: {max(lengths)}")
        print(f"  Mean: {np.mean(lengths):.1f}, Std: {np.std(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")

        unique, counts = np.unique(self.y_labels, return_counts=True)
        print(f"\nClass distribution:")
        for i, (cls, count) in enumerate(zip(unique, counts)):
            print(f"  Class {cls} ({self.class_names[cls]}): {count} ({count/len(self.y_labels)*100:.1f}%)")

        return {
            'lengths': lengths,
            'class_distribution': dict(zip(unique, counts))
        }

    def visualize_samples(self):
        """Visualize sample ECG signals from each class"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()

        for class_idx in range(4):
            class_indices = np.where(self.y_labels == class_idx)[0]
            if len(class_indices) > 0:
                sample_idx = class_indices[0]
                signal_data = self.X_data[sample_idx]

                axes[class_idx].plot(signal_data)
                axes[class_idx].set_title(f'Class {class_idx}: {self.class_names[class_idx]}')
                axes[class_idx].set_xlabel('Time (samples)')
                axes[class_idx].set_ylabel('Amplitude')
                axes[class_idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')

    def create_validation_split(self, test_size=0.2):
        """Create stratified validation split"""
        indices = np.arange(len(self.X_data))
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, stratify=self.y_labels, random_state=42
        )

        print(f"\nValidation split created:")
        print(f"  Training samples: {len(train_idx)}")
        print(f"  Validation samples: {len(val_idx)}")

        train_classes = np.bincount(self.y_labels[train_idx])
        val_classes = np.bincount(self.y_labels[val_idx])

        train_total = train_classes.sum()
        val_total = val_classes.sum()

        train_percent = train_classes / train_total * 100
        val_percent = val_classes / val_total * 100

        print("  Training class distribution:")
        for i, (count, percent) in enumerate(zip(train_classes, train_percent)):
            print(f"    Class {i}: {count} ({percent:.1f}%)")

        print("  Validation class distribution:")
        for i, (count, percent) in enumerate(zip(val_classes, val_percent)):
            print(f"    Class {i}: {count} ({percent:.1f}%)")


        return train_idx, val_idx

class ECGFeatureExtractor:
    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate

    def extract_statistical_features(self, signal):
        """Extract basic statistical features"""
        features = []
        features.extend([
            np.mean(signal),
            np.std(signal),
            np.var(signal),
            np.min(signal),
            np.max(signal),
            np.median(signal),
            np.percentile(signal, 25),
            np.percentile(signal, 75),
            len(signal)
        ])
        return features

    def extract_frequency_features(self, signal):
        """Extract frequency domain features"""
        fft_vals = np.abs(fft(signal))
        freqs = np.fft.fftfreq(len(signal), 1/self.sampling_rate)

        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_vals[:len(fft_vals)//2]

        features = []
        features.extend([
            np.mean(positive_fft),
            np.std(positive_fft),
            np.max(positive_fft),
            np.argmax(positive_fft),  
        ])

        return features

    def extract_all_features(self, signals):
        """Extract all features for a list of signals"""
        all_features = []
        for signal in signals:
            features = []
            features.extend(self.extract_statistical_features(signal))
            features.extend(self.extract_frequency_features(signal))
            all_features.append(features)

        return np.array(all_features)

class ECGAugmentor:
    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate

    def add_noise(self, signal, noise_factor=0.02):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_factor * np.std(signal), len(signal))
        return signal + noise

    def time_stretch(self, signal, stretch_factor=0.05):
        """Time stretching/compression"""
        factor = 1 + np.random.uniform(-stretch_factor, stretch_factor)
        indices = np.arange(0, len(signal), factor)
        indices = indices[indices < len(signal)].astype(int)
        return signal[indices]

    def amplitude_scale(self, signal, scale_factor=0.05):
        """Amplitude scaling"""
        factor = 1 + np.random.uniform(-scale_factor, scale_factor)
        return signal * factor

    def time_shift(self, signal, shift_factor=0.1):
        """Time shifting"""
        max_shift = int(len(signal) * shift_factor)
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal, shift)

    def augment_signal(self, signal, augmentation_prob=0.7):
        """Apply random augmentations"""
        # Ensure signal is a numpy array
        augmented = np.array(signal.copy())

        if np.random.random() < augmentation_prob:
            augmented = self.add_noise(augmented)

        if np.random.random() < augmentation_prob:
            augmented = self.time_stretch(augmented)

        if np.random.random() < augmentation_prob:
            augmented = self.amplitude_scale(augmented)

        if np.random.random() < augmentation_prob:
            augmented = self.time_shift(augmented)

        return augmented.tolist() # Convert back to list if needed later, or handle as numpy array

    def augment_dataset(self, X_data, y_labels, augmentation_factor=2):
        """Augment entire dataset"""
        augmented_X = []
        augmented_y = []

        for signal, label in zip(X_data, y_labels):
            augmented_X.append(signal)
            augmented_y.append(label)

            for _ in range(augmentation_factor):
                aug_signal = self.augment_signal(np.array(signal))
                augmented_X.append(aug_signal)
                augmented_y.append(label)

        return augmented_X, np.array(augmented_y)

class ECGDataset(Dataset):
    def __init__(self, X_data, y_labels, feature_extractor=None, augmentor=None, is_training=False):
        self.X_data = X_data
        self.y_labels = y_labels
        self.feature_extractor = feature_extractor
        self.augmentor = augmentor
        self.is_training = is_training

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        signal = np.array(self.X_data[idx], dtype=np.float32)
        label = self.y_labels[idx]

        if self.is_training and self.augmentor and np.random.random() < 0.5:
            signal = self.augmentor.augment_signal(signal)

        if len(signal) > 1:
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

        return torch.FloatTensor(signal), torch.LongTensor([label])[0]

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    signals, labels = zip(*batch)
    lengths = [len(s) for s in signals]

    max_len = max(lengths)
    padded_signals = torch.zeros(len(signals), max_len)

    for i, signal in enumerate(signals):
        padded_signals[i, :len(signal)] = signal

    return padded_signals, torch.stack(labels), lengths

class SimpleSTFTModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.n_fft = 1024
        self.hop_length = 128

        self.conv_blocks = nn.Sequential(
            self._conv_block(1, 64, dropout=0.3),
            self._conv_block(64, 128, dropout=0.3),
            self._conv_block(128, 256, dropout=0.4),
            self._conv_block(256, 512, dropout=0.4),
            self._conv_block(512, 256, dropout=0.4)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.rnn1 = self.rnn2 = None
        self.rnn_input_size = None

        # Fully connected classifier
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.bn_fc3 = nn.BatchNorm1d(256)

        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)

    def _conv_block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )

    def _compute_stft_batch(self, x):
        """Compute STFT for each sample individually and pad to uniform size"""
        batch_size = x.size(0)
        device = x.device
        stft_outputs = []

        for i in range(batch_size):
            signal_data = x[i].detach().cpu().numpy()
            _, _, stft = signal.stft(signal_data,
                                     nperseg=self.n_fft // 4,
                                     noverlap=self.n_fft // 8,
                                     window='hann')
            stft_mag = np.abs(stft)
            stft_outputs.append(torch.tensor(stft_mag, dtype=torch.float32))

        max_freq = max(s.size(0) for s in stft_outputs)
        max_time = max(s.size(1) for s in stft_outputs)
        padded = torch.zeros(batch_size, 1, max_freq, max_time, device=device)

        for i, s in enumerate(stft_outputs):
            s = s.to(device)
            padded[i, 0, :s.size(0), :s.size(1)] = s

        return torch.log2(padded + 1e-10)

    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        device = x.device

        # STFT -> Conv -> Pool
        x = self._compute_stft_batch(x)
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)

        # Reshape for RNN
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, x.size(3), -1)

        if self.rnn1 is None:
            self.rnn_input_size = x.size(-1)
            self.rnn1 = nn.LSTM(self.rnn_input_size, 256, batch_first=True, bidirectional=True).to(device)
            self.rnn2 = nn.LSTM(512, 256, batch_first=True, bidirectional=True).to(device)
            print(f"Initialized LSTM layers with input size: {self.rnn_input_size}")

        # BiLSTM layers
        x, _ = self.rnn1(x)
        x = self.dropout2(x)
        x, (h2, _) = self.rnn2(x)

        # Concatenate forward and backward final states
        x = torch.cat([h2[0], h2[1]], dim=1)  # shape: (batch, 512)

        # Fully connected classifier
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)

        return self.fc4(x)

class ECGTrainer:
    def __init__(self, model, device='cpu', name_prefix=''):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.5)
        self.name_prefix = name_prefix

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, targets, lengths) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data, lengths)
            loss = self.criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data, targets, lengths in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data, lengths)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = 100. * correct / total
        f1 = f1_score(all_targets, all_preds, average='macro')

        return total_loss / len(val_loader), accuracy, f1

    def train(self, train_loader, val_loader, epochs=100, patience=10):
        best_f1 = 0
        patience_counter, epoch = 0, 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            self.scheduler.step(val_loss)

            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                if os.path.exists('./Uni/AMLS'):
                    torch.save(self.model.state_dict(), './Uni/AMLS/' + self.name_prefix + self.model.__class__.__name__ + '_best_model.pth')
                else:
                    torch.save(self.model.state_dict(), self.name_prefix + self.model.__class__.__name__ + '_best_model.pth')

            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == epochs-1:
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}')

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if os.path.exists('./Uni/AMLS'):
            self.model.load_state_dict(torch.load('./Uni/AMLS/' + self.name_prefix + self.model.__class__.__name__  + '_best_model.pth'))
        else:
            self.model.load_state_dict(torch.load(self.name_prefix + self.model.__class__.__name__  + '_best_model.pth'))

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_f1': best_f1
        }

class Attention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key   = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, V)
        pooled = attended.mean(dim=1)
        return pooled, attn_weights

class ImprovedSTFTModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.n_fft = 512
        self.hop_length = 64

        self.conv_blocks = nn.Sequential(
            self._conv_block(1, 64, dropout=0.3),
            self._conv_block(64, 128, dropout=0.3),
            self._conv_block(128, 256, dropout=0.4),
            self._conv_block(256, 512, dropout=0.4),
            self._conv_block(512, 256, dropout=0.4)
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Lazy initialized RNN and Attention
        self.rnn1 = self.rnn2 = None
        self.attention = None
        self.rnn_input_size = None

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)

        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.bn_fc3 = nn.BatchNorm1d(256)

        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)

    def _conv_block(self, in_ch, out_ch, dropout):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout)
        )

    def _compute_stft_batch(self, x):
        """Compute log-magnitude STFT using torch.stft (B, T) → (B, 1, F, T)"""
        x = x.unsqueeze(1)  # (B, 1, T)
        window = torch.hann_window(window_length=self.n_fft).to(x.device)

        stft_result = torch.stft(
            x.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True
        )  # (B, F, T)

        mag = stft_result.abs()
        log_spec = torch.log1p(mag)
        return log_spec.unsqueeze(1)  # (B, 1, F, T)

    def forward(self, x, lengths=None):
        B = x.size(0)
        device = x.device

        # STFT → CNN → pool
        x = self._compute_stft_batch(x)
        x = self.conv_blocks(x)
        x = self.adaptive_pool(x)

        # Reshape for RNN
        x = x.permute(0, 3, 1, 2).contiguous().view(B, x.size(3), -1)

        if self.rnn1 is None:
            self.rnn_input_size = x.size(-1)
            self.rnn1 = nn.LSTM(self.rnn_input_size, 256, batch_first=True, bidirectional=True).to(device)
            self.rnn2 = nn.LSTM(512, 256, batch_first=True, bidirectional=True).to(device)
            self.attention = Attention(input_dim=512).to(device)
            print(f"Initialized RNNs and Attention with input size: {self.rnn_input_size}")

        x, _ = self.rnn1(x)
        x = self.dropout2(x)
        x, _ = self.rnn2(x)
        x, _ = self.attention(x)

        # FC classification
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout2(x)

        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.dropout3(x)

        return self.fc4(x)

class ECGAugmentor:
    def __init__(self, sampling_rate=300):
        self.sampling_rate = sampling_rate

    def add_noise(self, signal, noise_factor=0.02):
        """Add Gaussian noise - reduced default noise"""
        noise = np.random.normal(0, noise_factor * np.std(signal), len(signal))
        return signal + noise

    def time_stretch(self, signal, stretch_factor=0.05):
        """Improved time stretching using interpolation"""
        factor = 1 + np.random.uniform(-stretch_factor, stretch_factor)
        original_length = len(signal)

        original_indices = np.arange(original_length)
        new_length = int(original_length / factor)
        new_indices = np.linspace(0, original_length - 1, new_length)

        interp_func = interp1d(original_indices, signal, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        stretched = interp_func(new_indices)

        if len(stretched) < original_length:
            pad_length = original_length - len(stretched)
            stretched = np.pad(stretched, (0, pad_length), mode='edge')
        elif len(stretched) > original_length:
            stretched = stretched[:original_length]

        return stretched

    def amplitude_scale(self, signal, scale_factor=0.05):
        """Reduced amplitude scaling"""
        factor = 1 + np.random.uniform(-scale_factor, scale_factor)
        return signal * factor

    def time_shift(self, signal, shift_factor=0.05):
        """Reduced time shifting"""
        max_shift = int(len(signal) * shift_factor)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(signal, shift)

    def baseline_wander(self, signal, amplitude=0.1):
        """Add baseline wander (common ECG artifact)"""
        length = len(signal)
        # Create low frequency sinusoidal baseline wander
        freq = np.random.uniform(0.1, 0.5)  # Hz
        t = np.arange(length) / self.sampling_rate
        wander = amplitude * np.sin(2 * np.pi * freq * t)
        return signal + wander

    def powerline_interference(self, signal, amplitude=0.05):
        """Add 50/60 Hz powerline interference"""
        length = len(signal)
        t = np.arange(length) / self.sampling_rate
        freq = np.random.choice([50, 60])
        interference = amplitude * np.sin(2 * np.pi * freq * t)
        return signal + interference

    def muscle_artifact(self, signal, amplitude=0.03, prob=0.3):
        """Add muscle artifact (high frequency noise in random segments)"""
        if np.random.random() > prob:
            return signal

        result = signal.copy()
        segment_length = int(len(signal) * 0.1)  # 10% of signal length
        start_idx = np.random.randint(0, len(signal) - segment_length)
        muscle_noise = np.random.normal(0, amplitude, segment_length)
        result[start_idx:start_idx + segment_length] += muscle_noise

        return result

    def augment_signal(self, signal, augmentation_prob=0.5):
        """Apply random augmentations with lower probability"""
        augmented = np.array(signal, dtype=np.float32)

        augmentations = [
            (self.add_noise, 0.3),
            (self.time_stretch, 0.2),
            (self.amplitude_scale, 0.3),
            (self.time_shift, 0.2),
            (self.baseline_wander, 0.2),
            (self.powerline_interference, 0.15),
            (self.muscle_artifact, 0.15)
        ]

        for aug_func, prob in augmentations:
            if np.random.random() < prob:
                augmented = aug_func(augmented)

        return augmented

    def augment_dataset(self, X_data, y_labels, augmentation_factor=1):
        """Augment dataset with better class balance handling"""
        y_labels = np.array(y_labels)
        unique_classes, class_counts = np.unique(y_labels, return_counts=True)
        max_count = np.max(class_counts)

        augmented_X = []
        augmented_y = []

        augmented_X.extend(X_data)
        augmented_y.extend(y_labels)

        for class_label in unique_classes:
            class_indices = np.where(y_labels == class_label)[0]
            class_count = len(class_indices)

            if class_count < max_count:
                target_augmentations = int((max_count - class_count) * augmentation_factor)
            else:
                target_augmentations = int(class_count * augmentation_factor * 0.3)

            for _ in range(target_augmentations):
                idx = np.random.choice(class_indices)
                original_signal = X_data[idx]
                augmented_signal = self.augment_signal(original_signal)
                augmented_X.append(augmented_signal)
                augmented_y.append(class_label)

        return augmented_X, np.array(augmented_y)

    def pad_or_truncate(self, signals, target_length=None):
        """Pad or truncate signals to same length"""
        if target_length is None:
            lengths = [len(signal) for signal in signals]
            target_length = int(np.median(lengths))

        processed_signals = []
        for signal in signals:
            if len(signal) < target_length:
                padded = np.pad(signal, (0, target_length - len(signal)), mode='constant')
                processed_signals.append(padded)
            elif len(signal) > target_length:
                processed_signals.append(signal[:target_length])
            else:
                processed_signals.append(signal)

        return np.array(processed_signals)

    def validate_augmentation(self, original_signal, augmented_signal, max_change=0.3):
        """Validate that augmentation doesn't change signal too much"""
        correlation = np.corrcoef(original_signal, augmented_signal)[0, 1]
        orig_amplitude = np.max(original_signal) - np.min(original_signal)
        aug_amplitude = np.max(augmented_signal) - np.min(augmented_signal)
        amplitude_change = abs(aug_amplitude - orig_amplitude) / orig_amplitude

        return correlation > 0.7 and amplitude_change < max_change

class ECGDataReducer:
    def __init__(self):
        pass

    def stratified_sample(self, X_data, y_labels, reduction_ratio=0.25):
        """Stratified random sampling"""
        indices = np.arange(len(X_data))
        _, sampled_indices = train_test_split(
            indices, test_size=reduction_ratio, stratify=y_labels, random_state=42
        )

        return [X_data[i] for i in sampled_indices], y_labels[sampled_indices]

    def coreset_selection(self, X_data, y_labels, reduction_ratio=0.25):
        """Simple coreset selection based on feature diversity"""
        feature_extractor = ECGFeatureExtractor()
        features = feature_extractor.extract_all_features(X_data)
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        selected_indices = []
        unique_classes = np.unique(y_labels)

        for cls in unique_classes:
            class_indices = np.where(y_labels == cls)[0]
            class_features = features_normalized[class_indices]
            n_select = max(1, int(len(class_indices) * reduction_ratio))

            if len(class_indices) <= n_select:
                selected_indices.extend(class_indices)
            else:
                from sklearn.cluster import KMeans
                if n_select > 1:
                    kmeans = KMeans(n_clusters=n_select, random_state=42)
                    kmeans.fit(class_features)

                    for center in kmeans.cluster_centers_:
                        distances = np.linalg.norm(class_features - center, axis=1)
                        closest_idx = class_indices[np.argmin(distances)]
                        if closest_idx not in selected_indices:
                            selected_indices.append(closest_idx)
                else:
                    mean_features = np.mean(class_features, axis=0)
                    distances = np.linalg.norm(class_features - mean_features, axis=1)
                    selected_indices.append(class_indices[np.argmin(distances)])

        return [X_data[i] for i in selected_indices], y_labels[selected_indices]

    def compress_signal(self, signal, compression_ratio=0.5):
        """Simple signal compression using FFT"""
        fft_signal = fft(signal)
        n_keep = int(len(fft_signal) * compression_ratio)
        compressed_fft = np.zeros_like(fft_signal)
        compressed_fft[:n_keep//2] = fft_signal[:n_keep//2]
        compressed_fft[-n_keep//2:] = fft_signal[-n_keep//2:]

        return np.real(ifft(compressed_fft))

def generate_base_predictions(model, test_loader, device, pre=''):
  model.eval()
  base_predictions = []
  with torch.no_grad():
      for data, _, lengths in test_loader:
          data = data.to(device)
          outputs = model(data, lengths)
          _, predicted = outputs.max(1)
          base_predictions.extend(predicted.cpu().numpy())

  adr = pre + model.__class__.__name__ + '_base.csv'
  if os.path.exists('./Uni/AMLS'):
    adr = './Uni/AMLS/' + pre + model.__class__.__name__ + '_base.csv'
  pd.DataFrame({'label': base_predictions}).to_csv(adr, index=False)
  print(f"Base predictions saved to {adr}")

def compress_dataset(X_data, y_labels, output_path="compressed_data.zip"):
    """Compress dataset using custom format"""

    binary_data = bytearray()

    for signal in X_data:
        binary_data.extend(struct.pack('i', len(signal)))
        for val in signal:
            binary_data.extend(struct.pack('h', int(val)))

    labels_df = pd.DataFrame({'label': y_labels})
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('data.bin', binary_data)
        zf.writestr('labels.csv', labels_df.to_csv(index=False))

    print(f"Compressed dataset saved to {output_path}")
    print(f"Original size: {len(binary_data)} bytes")
    print(f"Compressed size: {os.path.getsize(output_path)} bytes")
    print(f"Compression ratio: {os.path.getsize(output_path)/len(binary_data):.3f}")


def train_and_evaluate_reduced_models(X_train_split, y_train_split, X_val_split, y_val_split,
                                    X_test, device='cpu'):
    """
    Train models on reduced datasets and evaluate performance
    """

    reducer = ECGDataReducer()
    reduction_ratios = [0.1, 0.25, 0.5, 1.0]  # Include 100% for baseline
    results = {
        'reduction_ratio': [],
        'method': [],
        'f1_score': [],
        'accuracy': [],
        'train_samples': [],
        'val_f1': [],
        'val_accuracy': []
    }

    print("Training models on reduced datasets...")
    print("=" * 60)

    for ratio in reduction_ratios:
        print(f"\nTraining with {ratio*100}% of data...")

        if ratio == 1.0:
            X_train_reduced = X_train_split
            y_train_reduced = y_train_split
            method = 'full'
        else:
            for method in ['stratified', 'coreset']:
                print(f"  Method: {method}")

                if method == 'stratified':
                    X_train_reduced, y_train_reduced = reducer.stratified_sample(
                        X_train_split, y_train_split, ratio
                    )
                else:  # coreset
                    X_train_reduced, y_train_reduced = reducer.coreset_selection(
                        X_train_split, y_train_split, ratio
                    )

                print(f"    Training samples: {len(X_train_reduced)}")
                model, train_history = train_model_on_reduced_data(
                    X_train_reduced, y_train_reduced, X_val_split, y_val_split,
                    device, f"reduced_{ratio}_{method}"
                )

                val_f1, val_acc = evaluate_model_on_validation(
                    model, X_val_split, y_val_split, device
                )

                results['reduction_ratio'].append(ratio)
                results['method'].append(method)
                results['f1_score'].append(train_history['best_f1'])
                results['accuracy'].append(val_acc)
                results['train_samples'].append(len(X_train_reduced))
                results['val_f1'].append(val_f1)
                results['val_accuracy'].append(val_acc)

                print(f"    Validation F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")
                if ratio == 0.25 and method == 'coreset':
                    print("    Generating predictions for reduced.csv...")
                    generate_reduced_predictions(model, X_test, device)

        if ratio == 1.0:
            print(f"  Method: full dataset")
            model, train_history = train_model_on_reduced_data(
                X_train_reduced, y_train_reduced, X_val_split, y_val_split,
                device, "full"
            )

            val_f1, val_acc = evaluate_model_on_validation(
                model, X_val_split, y_val_split, device
            )

            results['reduction_ratio'].append(ratio)
            results['method'].append('full')
            results['f1_score'].append(train_history['best_f1'])
            results['accuracy'].append(val_acc)
            results['train_samples'].append(len(X_train_reduced))
            results['val_f1'].append(val_f1)
            results['val_accuracy'].append(val_acc)

            print(f"    Validation F1: {val_f1:.4f}, Accuracy: {val_acc:.4f}")

    return results

def train_model_on_reduced_data(X_train, y_train, X_val, y_val, device, name_prefix):
    """
    Train a model on reduced data
    """
    augmentor = ECGAugmentor()

    train_dataset = ECGDataset(X_train, y_train, augmentor=augmentor, is_training=True)
    val_dataset = ECGDataset(X_val, y_val, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = ImprovedSTFTModel(num_classes=4)
    trainer = ECGTrainer(model, device=device, name_prefix=name_prefix + '_')
    history = trainer.train(train_loader, val_loader, epochs=80, patience=40)

    return model, history

def evaluate_model_on_validation(model, X_val, y_val, device):
    """
    Evaluate model on validation set
    """
    val_dataset = ECGDataset(X_val, y_val, is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, targets, lengths in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data, lengths)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    f1 = f1_score(all_targets, all_preds, average='macro')
    accuracy = accuracy_score(all_targets, all_preds)

    return f1, accuracy

def generate_reduced_predictions(model, X_test, device):
    """
    Generate predictions for the test set using reduced model
    """
    test_dataset = ECGDataset(X_test, [0] * len(X_test), is_training=False)  # Dummy labels
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    model.eval()
    predictions = []

    with torch.no_grad():
        for data, _, lengths in test_loader:
            data = data.to(device)
            outputs = model(data, lengths)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())

    output_path = 'reduced.csv'
    if os.path.exists('./Uni/AMLS'):
        output_path = './Uni/AMLS/reduced.csv'

    pd.DataFrame({'label': predictions}).to_csv(output_path, index=False)
    print(f"Reduced predictions saved to {output_path}")

def create_results_plot(results):
    """
    Create visualization of results across different data sizes
    """
    df = pd.DataFrame(results)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    for method in ['stratified', 'coreset', 'full']:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax1.plot(method_data['reduction_ratio'] * 100, method_data['val_f1'],
                    marker='o', linewidth=2, markersize=8, label=method.capitalize())

    ax1.set_xlabel('Dataset Size (%)')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('F1 Score vs Dataset Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 105)

    for method in ['stratified', 'coreset', 'full']:
        method_data = df[df['method'] == method]
        if len(method_data) > 0:
            ax2.plot(method_data['reduction_ratio'] * 100, method_data['val_accuracy'],
                    marker='s', linewidth=2, markersize=8, label=method.capitalize())

    ax2.set_xlabel('Dataset Size (%)')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Dataset Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 105)

    plt.tight_layout()
    plt.savefig('data_reduction_results.png', dpi=300, bbox_inches='tight')
    if os.path.exists('./Uni/AMLS'):
        plt.savefig('./Uni/AMLS/data_reduction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\n" + "="*60)
    print("SUMMARY OF DATA REDUCTION RESULTS")
    print("="*60)

    summary_df = df.pivot_table(
        values=['val_f1', 'val_accuracy'],
        index='reduction_ratio',
        columns='method',
        aggfunc='mean'
    )

    print("\nF1 Scores:")
    print(summary_df['val_f1'].round(4))
    print("\nAccuracy:")
    print(summary_df['val_accuracy'].round(4))

    print("\n" + "="*60)
    print("CORESET vs STRATIFIED COMPARISON")
    print("="*60)

    for ratio in [0.1, 0.25, 0.5]:
        strat_f1 = df[(df['reduction_ratio'] == ratio) & (df['method'] == 'stratified')]['val_f1'].values
        coreset_f1 = df[(df['reduction_ratio'] == ratio) & (df['method'] == 'coreset')]['val_f1'].values

        if len(strat_f1) > 0 and len(coreset_f1) > 0:
            improvement = coreset_f1[0] - strat_f1[0]
            print(f"{ratio*100:2.0f}% data: Coreset F1={coreset_f1[0]:.4f}, Stratified F1={strat_f1[0]:.4f}, "
                  f"Improvement={improvement:+.4f}")

def create_compressed_datasets(X_train_split, y_train_split, reduction_ratios=[0.1, 0.25, 0.5]):
    """
    Create compressed datasets for different reduction ratios
    """

    reducer = ECGDataReducer()

    print("Creating compressed datasets...")
    print("=" * 40)

    for ratio in reduction_ratios:
        X_reduced, y_reduced = reducer.coreset_selection(X_train_split, y_train_split, ratio)
        output_path = f'reduced_data_{int(ratio*100)}percent.zip'
        if os.path.exists('./Uni/AMLS'):
            output_path = f'./Uni/AMLS/reduced_data_{int(ratio*100)}percent.zip'

        compress_dataset(X_reduced, y_reduced, output_path)

        print(f"Created {output_path} with {len(X_reduced)} samples")

def read_compressed_dataset(path):
    """
    Read compressed dataset
    """

    X_data = []

    with zipfile.ZipFile(path, 'r') as zf:
        inner_path = os.path.basename(path).split('.')[0]
        try:
            with zf.open('data.bin', 'r') as f:
                read_binary_from(X_data, f)
        except:
            with zf.open(f'{inner_path}.bin', 'r') as f:
                read_binary_from(X_data, f)

        try:
            with zf.open('labels.csv', 'r') as f:
                labels_df = pd.read_csv(f)
                y_labels = labels_df['label'].values
        except:
            with zf.open(f'{inner_path}_labels.csv', 'r') as f:
                labels_df = pd.read_csv(f)
                y_labels = labels_df['label'].values

    return X_data, y_labels


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].upper() in ["AUGMENTED", "REDUCED", "SIMPLE", "IMPROVED"]:
        print("ECG Time Series Classification Pipeline")
        print("=" * 50)
        X_train = read_zip_binary('./Uni/AMLS/X_train.zip')
        y_train = load_labels('./Uni/AMLS/y_train.csv')
        X_test = read_zip_binary('./Uni/AMLS/X_test.zip')

        print(f"Loaded {len(X_train)} training samples and {len(X_test)} test samples")

        print("\n" + "="*20 + " TASK 1: DATA EXPLORATION " + "="*20)

        explorer = ECGDataExplorer(X_train, y_train)
        stats = explorer.analyze_data()
        explorer.visualize_samples()
        train_idx, val_idx = explorer.create_validation_split()

        X_train_split = [X_train[i] for i in train_idx]
        y_train_split = y_train[train_idx]
        X_val_split = [X_train[i] for i in val_idx]
        y_val_split = y_train[val_idx]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        train_dataset = ECGDataset(X_train_split, y_train_split, is_training=True)
        val_dataset = ECGDataset(X_val_split, y_val_split, is_training=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        test_dataset = ECGDataset(X_test, np.zeros(len(X_test)), is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        arg = sys.argv[1].upper()
        if arg == "AUGMENTED":
            print("\n" + "TASK 3: DATA AUGMENTATION (ImprovedSTFT)")
            augmentor = ECGAugmentor()
            X_train_aug, y_train_aug = augmentor.augment_dataset(X_train_split, y_train_split, augmentation_factor=0.5)
            print(f"Augmented dataset size: {len(X_train_aug)} (original: {len(X_train_split)})")
            train_dataset_aug = ECGDataset(X_train_aug, y_train_aug, augmentor=augmentor, is_training=True)
            train_loader_aug = DataLoader(train_dataset_aug, batch_size=32, shuffle=True, collate_fn=collate_fn)
            print("Training STFT...")
            model_aug = ImprovedSTFTModel(num_classes=4)
            trainer_aug = ECGTrainer(model_aug, device=device, name_prefix='augment_')
            print(f'Model initialized with {sum(p.numel() for p in model_aug.parameters())} parameters')
            print("Training with augmented data...")
            history_aug = trainer_aug.train(train_loader_aug, val_loader, epochs=100, patience=50)
            test_dataset = ECGDataset(X_test, np.zeros(len(X_test)), is_training=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            generate_base_predictions(model_aug, test_loader, device, pre='augment_')
            plot_training_history(history_aug, model_name="AugmentedSTFTModel", save_path="training_augmented_history.png")
        elif arg == "REDUCED":
            print("\n" + "TASK: DATA REDUCTION ")
            results = train_and_evaluate_reduced_models(
                X_train_split, y_train_split, X_val_split, y_val_split, X_test, device
            )
            create_results_plot(results)
            create_compressed_datasets(X_train_split, y_train_split)
            results_df = pd.DataFrame(results)
            output_path = '/content/drive/MyDrive/AMLS/data_reduction_results.csv'
            if os.path.exists('./Uni/AMLS'):
                output_path = './Uni/AMLS/data_reduction_results.csv'
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
            print("\n" + "="*60)
            print("DATA REDUCTION EXPERIMENT COMPLETED")
            print("="*60)
            print("Files created:")
            print("- reduced.csv: Test predictions using 25% coreset model")
            print("- data_reduction_results.png: Performance comparison plot")
            print("- data_reduction_results.csv: Detailed results")
            print("- reduced_data_*percent.zip: Compressed datasets")
        elif arg == "SIMPLE":
            print("\n" + " TASK 2: MODELING AND TUNING ")
            print("Training SimpleSTFT...")
            simple_model = SimpleSTFTModel(num_classes=4)
            trainer = ECGTrainer(simple_model, device=device)
            print(f'simple_model initialized with {sum(p.numel() for p in simple_model.parameters())} parameters')
            print("Starting training...")
            history = trainer.train(train_loader, val_loader, epochs=70, patience=40)
            test_dataset = ECGDataset(X_test, np.zeros(len(X_test)), is_training=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
            generate_base_predictions(simple_model, test_loader, device)
            plot_training_history(history, model_name="SimpleSTFTModel", save_path="training_simple_model_history.png")
        elif arg == "IMPROVED":
            print("\n" + "TASK 2: MODELING AND TUNING ")
            print("Training ImprovedSTFT...")
            improved_model = ImprovedSTFTModel(num_classes=4)
            trainer = ECGTrainer(improved_model, device=device)
            print(f'Model initialized with {sum(p.numel() for p in improved_model.parameters())} parameters')
            print("Starting training...")
            history_improved = trainer.train(train_loader, val_loader, epochs=90, patience=40)
            generate_base_predictions(improved_model, test_loader, device)
            plot_training_history(history_improved, model_name="ImprovedSTFTModel", save_path="training_improved_model_history.png")
    else:
        print("Usage: ./AMLS.py [IMPROVED|SIMPLE|AUGMENTED|REDUCED]")
        print("  IMPROVED  : Train and evaluate the ImprovedSTFTModel (baseline).")
        print("  SIMPLE    : Train and evaluate the SimpleSTFTModel (baseline).")
        print("  AUGMENTED : Train and evaluate the ImprovedSTFTModel with data augmentation.")
        print("  REDUCED   : Run data reduction experiments and generate reduced.csv, plots, and compressed datasets.")