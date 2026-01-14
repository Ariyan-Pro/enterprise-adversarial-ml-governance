"""
Models module for adversarial ML security suite
"""
from .base.mnist_cnn import MNISTCNN, create_mnist_cnn

__all__ = ['MNISTCNN', 'create_mnist_cnn']
