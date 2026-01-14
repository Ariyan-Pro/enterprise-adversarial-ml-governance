import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST_CNN(nn.Module):
    """
    Enhanced CNN for MNIST classification
    Matches the saved model architecture: conv1=16, conv2=32
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.2):
        super(MNIST_CNN, self).__init__()
        
        # Feature extraction layers - MUST MATCH SAVED MODEL
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Changed from 32 to 16
        self.bn1 = nn.BatchNorm2d(16)  # Changed from 32 to 16
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Changed from 64 to 32
        self.bn2 = nn.BatchNorm2d(32)  # Changed from 64 to 32
        
        # Fully connected layers
        # Input size calculation: 32 filters * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Changed from 64*7*7=3136 to 32*7*7=1568
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Factory function for backward compatibility
def create_mnist_cnn(num_classes=10, dropout_rate=0.2):
    """Factory function to create MNIST CNN"""
    return MNIST_CNN(num_classes=num_classes, dropout_rate=dropout_rate)


