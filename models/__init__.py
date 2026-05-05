"""
Models module for adversarial ML security suite
"""
from .base.mnist_cnn import MNIST_CNN, create_mnist_cnn

__all__ = ['MNIST_CNN', 'create_mnist_cnn']
