#!/usr/bin/env python
"""
Data Exploration Notebook - Convert to .ipynb with: jupyter nbconvert --to notebook --execute 01_data_exploration.py
"""

# %% [markdown]
# # Data Exploration - MNIST Dataset

# %%
import sys
sys.path.insert(0, '..')

from utils.dataset_utils import load_mnist, get_dataset_stats
from utils.visualization import setup_plotting

import matplotlib.pyplot as plt
import numpy as np
import torch

# %% [markdown]
# ## Load Dataset

# %%
# Load MNIST
train_set, test_set = load_mnist()

print(f"Training samples: {len(train_set)}")
print(f"Test samples: {len(test_set)}")

# %% [markdown]
# ## Dataset Statistics

# %%
# Get statistics
train_stats = get_dataset_stats(train_set)
test_stats = get_dataset_stats(test_set)

print("Training set statistics:")
for key, value in train_stats.items():
    print(f"  {key}: {value}")

print("\nTest set statistics:")
for key, value in test_stats.items():
    print(f"  {key}: {value}")

# %% [markdown]
# ## Visualize Samples

# %%
# Plot sample images
setup_plotting()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i in range(10):
    # Find first example of each class
    indices = [j for j, (_, label) in enumerate(train_set) if label == i]
    img, label = train_set[indices[0]]
    
    axes[i].imshow(img.squeeze(), cmap='gray')
    axes[i].set_title(f'Digit: {label}')
    axes[i].axis('off')

plt.suptitle('MNIST Sample Images (One per Class)', fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Class Distribution

# %%
# Plot class distribution
train_labels = [label for _, label in train_set]
test_labels = [label for _, label in test_set]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Training distribution
ax1.hist(train_labels, bins=10, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Digit')
ax1.set_ylabel('Count')
ax1.set_title('Training Set Class Distribution')
ax1.set_xticks(range(10))

# Test distribution
ax2.hist(test_labels, bins=10, edgecolor='black', alpha=0.7)
ax2.set_xlabel('Digit')
ax2.set_ylabel('Count')
ax2.set_title('Test Set Class Distribution')
ax2.set_xticks(range(10))

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Pixel Intensity Analysis

# %%
# Analyze pixel intensities
train_data = torch.stack([img for img, _ in train_set])
test_data = torch.stack([img for img, _ in test_set])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Training pixel histogram
ax1.hist(train_data.flatten().numpy(), bins=50, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Pixel Intensity')
ax1.set_ylabel('Frequency')
ax1.set_title('Training Set Pixel Distribution')
ax1.grid(True, alpha=0.3)

# Test pixel histogram
ax2.hist(test_data.flatten().numpy(), bins=50, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Pixel Intensity')
ax2.set_ylabel('Frequency')
ax2.set_title('Test Set Pixel Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
