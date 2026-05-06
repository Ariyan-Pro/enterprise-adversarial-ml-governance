"""
🏗️ ENTERPRISE MODEL ROUTER - Multi-domain model management
Supported simultaneously: vision, tabular, text classifiers
"""
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import importlib.util
import torch
import torch.nn as nn

class ModelDomain(Enum):
    VISION = "vision"
    TABULAR = "tabular"
    TEXT = "text"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

class ModelState(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    QUARANTINED = "quarantined"
    RETIRED = "retired"

@dataclass
class ModelMetadata:
    """Complete model metadata"""
    model_id: str
    name: str
    version: str
    domain: ModelDomain
    state: ModelState
    architecture: str
    input_shape: List[int]
    output_classes: int
    accuracy: float
    robustness_score: float
    training_data: str
    created_at: str
    updated_at: str
    owner: str
    dependencies: List[str]
    deployment_config: Dict[str, Any]

@dataclass 
class ModelInstance:
    """Loaded model instance"""
    metadata: ModelMetadata
    model: nn.Module
    preprocessor: Any
    postprocessor: Any
    robustness_history: List[Dict[str, Any]]

class EnterpriseModelRouter:
    """Enterprise model routing and lifecycle management"""
    
    def __init__(self, registry_path: str = "models/registry/models.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load_registry()
        self.loaded_models: Dict[str, ModelInstance] = {}
        self.active_models: Dict[ModelDomain, str] = {}  # domain -> active_model_id
        
    def route(self, request: Dict[str, Any]) -> ModelInstance:
        """Route request to appropriate model"""
        
        # Determine domain from request
        domain = self._detect_domain(request)
        
        # Get active model for domain
        if domain not in self.active_models:
            raise ValueError(f"No active model configured for domain: {domain}")
        
        model_id = self.active_models[domain]
        
        # Load model if not already loaded
        if model_id not in self.loaded_models:
            self.loaded_models[model_id] = self._load_model(model_id)
        
        return self.loaded_models[model_id]
    
    def register_model(self, model_info: Dict[str, Any]) -> str:
        """Register a new model in the enterprise registry"""
        
        # Generate unique model ID
        model_id = f"{model_info['domain']}_{model_info['name']}_v{model_info['version']}"
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            name=model_info["name"],
            version=model_info["version"],
            domain=ModelDomain(model_info["domain"]),
            state=ModelState.DEVELOPMENT,  # Start in development
            architecture=model_info.get("architecture", "unknown"),
            input_shape=model_info["input_shape"],
            output_classes=model_info["output_classes"],
            accuracy=model_info.get("accuracy", 0.0),
            robustness_score=model_info.get("robustness_score", 0.0),
            training_data=model_info.get("training_data", "unknown"),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            owner=model_info.get("owner", "unknown"),
            dependencies=model_info.get("dependencies", []),
            deployment_config=model_info.get("deployment_config", {})
        )
        
        # Save model files
        model_path = Path(f"models/{metadata.domain.value}/{model_id}")
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = model_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Save model weights if provided
        if "model_weights" in model_info:
            weights_path = model_path / "model.pth"
            torch.save(model_info["model_weights"], weights_path)
        
        # Update registry
        self.registry[model_id] = asdict(metadata)
        self._save_registry()
        
        return model_id
    
    def promote_model(self, model_id: str, target_state: ModelState) -> bool:
        """Promote model through lifecycle states"""
        if model_id not in self.registry:
            raise ValueError(f"Model not found: {model_id}")
        
        current_state = ModelState(self.registry[model_id]["state"])
        
        # Check valid state transition
        valid_transitions = {
            ModelState.DEVELOPMENT: [ModelState.STAGING, ModelState.QUARANTINED],
            ModelState.STAGING: [ModelState.PRODUCTION, ModelState.DEVELOPMENT, ModelState.QUARANTINED],
            ModelState.PRODUCTION: [ModelState.STAGING, ModelState.QUARANTINED, ModelState.RETIRED],
            ModelState.QUARANTINED: [ModelState.DEVELOPMENT, ModelState.RETIRED],
            ModelState.RETIRED: []  # Final state
        }
        
        if target_state not in valid_transitions[current_state]:
            raise ValueError(
                f"Cannot transition from {current_state.value} to {target_state.value}. "
                f"Valid transitions: {[s.value for s in valid_transitions[current_state]]}"
            )
        
        # Update state
        self.registry[model_id]["state"] = target_state.value
        self.registry[model_id]["updated_at"] = datetime.now().isoformat()
        
        # If promoting to production, set as active for domain
        if target_state == ModelState.PRODUCTION:
            domain = ModelDomain(self.registry[model_id]["domain"])
            self.active_models[domain] = model_id
        
        self._save_registry()
        return True
    
    def list_models(self, domain: Optional[ModelDomain] = None) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by domain"""
        if domain:
            return [
                model for model_id, model in self.registry.items()
                if ModelDomain(model["domain"]) == domain
            ]
        return list(self.registry.values())
    
    def _detect_domain(self, request: Dict[str, Any]) -> ModelDomain:
        """Detect model domain from request"""
        data = request.get("data", {})
        
        # Check for explicit domain
        if "domain" in request:
            try:
                return ModelDomain(request["domain"])
            except ValueError:
                pass
        
        # Infer from data structure
        if "image" in data or "pixels" in data:
            return ModelDomain.VISION
        elif "features" in data or "columns" in data:
            return ModelDomain.TABULAR
        elif "text" in data or "tokens" in data:
            return ModelDomain.TEXT
        elif "audio" in data or "waveform" in data:
            return ModelDomain.AUDIO
        else:
            # Default to vision for MNIST compatibility
            return ModelDomain.VISION
    
    def _load_model(self, model_id: str) -> ModelInstance:
        """Load model from registry"""
        if model_id not in self.registry:
            raise ValueError(f"Model not found: {model_id}")
        
        metadata_dict = self.registry[model_id]
        metadata = ModelMetadata(**metadata_dict)
        
        # Load model based on architecture
        model = self._instantiate_model(metadata)
        
        # Load weights
        model_path = Path(f"models/{metadata.domain.value}/{model_id}")
        weights_path = model_path / "model.pth"
        
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        
        model.eval()
        
        # Create pre/post processors
        preprocessor = self._create_preprocessor(metadata)
        postprocessor = self._create_postprocessor(metadata)
        
        # Load robustness history
        robustness_path = model_path / "robustness.json"
        robustness_history = []
        if robustness_path.exists():
            with open(robustness_path, "r") as f:
                robustness_history = json.load(f)
        
        return ModelInstance(
            metadata=metadata,
            model=model,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            robustness_history=robustness_history
        )
    
    def _instantiate_model(self, metadata: ModelMetadata) -> nn.Module:
        """Instantiate model based on architecture and domain"""
        domain = metadata.domain
        architecture = metadata.architecture.lower()
        
        # VISION domain models
        if domain == ModelDomain.VISION:
            return self._load_vision_model(architecture, metadata)
        
        # TABULAR domain models
        elif domain == ModelDomain.TABULAR:
            return self._load_tabular_model(architecture, metadata)
        
        # TEXT domain models
        elif domain == ModelDomain.TEXT:
            return self._load_text_model(architecture, metadata)
        
        # AUDIO domain models
        elif domain == ModelDomain.AUDIO:
            return self._load_audio_model(architecture, metadata)
        
        else:
            raise NotImplementedError(f"Model loading not implemented for domain: {domain}")
    
    def _load_vision_model(self, architecture: str, metadata: ModelMetadata) -> nn.Module:
        """Load vision models (CNN, ResNet, etc.)"""
        try:
            from models.base.mnist_cnn import MNISTCNN
            return MNISTCNN(num_classes=metadata.output_classes)
        except ImportError:
            pass
        
        # Fallback to simple CNN
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.conv1(x)
                x = nn.functional.relu(x)
                x = self.conv2(x)
                x = nn.functional.relu(x)
                x = nn.functional.max_pool2d(x, 2)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = nn.functional.relu(x)
                x = self.fc2(x)
                return x
        
        return SimpleCNN(num_classes=metadata.output_classes)
    
    def _load_tabular_model(self, architecture: str, metadata: ModelMetadata) -> nn.Module:
        """Load tabular models (MLP, XGBoost wrapper, etc.)"""
        input_features = metadata.input_shape[0] if metadata.input_shape else 10
        num_classes = metadata.output_classes
        
        # Check for specific architectures
        if 'mlp' in architecture or 'dense' in architecture or 'feedforward' in architecture:
            class TabularMLP(nn.Module):
                def __init__(self, input_size=input_features, hidden_size=128, num_classes=num_classes):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_size),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_size, hidden_size // 2),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_size // 2),
                        nn.Dropout(0.3),
                        nn.Linear(hidden_size // 2, num_classes)
                    )
                
                def forward(self, x):
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    return self.network(x)
            
            return TabularMLP()
        
        # Default to MLP for tabular data
        class SimpleTabularNet(nn.Module):
            def __init__(self, input_size=input_features, num_classes=num_classes):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes)
                )
            
            def forward(self, x):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return self.network(x)
        
        return SimpleTabularNet()
    
    def _load_text_model(self, architecture: str, metadata: ModelMetadata) -> nn.Module:
        """Load text models (LSTM, Transformer, BERT wrapper, etc.)"""
        vocab_size = metadata.input_shape[0] if metadata.input_shape else 10000
        embedding_dim = 128
        hidden_size = 256
        num_classes = metadata.output_classes
        
        # LSTM-based text classifier
        if 'lstm' in architecture or 'rnn' in architecture or 'gru' in architecture:
            class TextLSTM(nn.Module):
                def __init__(self, vocab_size=vocab_size, embed_dim=embedding_dim, 
                             hidden_size=hidden_size, num_classes=num_classes):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                    self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True, 
                                       bidirectional=True, dropout=0.3)
                    self.fc = nn.Linear(hidden_size * 2, num_classes)
                    self.dropout = nn.Dropout(0.5)
                
                def forward(self, x):
                    # x shape: (batch, seq_len)
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    embedded = self.dropout(self.embedding(x))
                    lstm_out, (hidden, cell) = self.lstm(embedded)
                    # Concatenate final forward and backward hidden states
                    hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                    return self.fc(self.dropout(hidden_cat))
            
            return TextLSTM()
        
        # Transformer-based text classifier
        if 'transformer' in architecture or 'bert' in architecture or 'attention' in architecture:
            class TextTransformer(nn.Module):
                def __init__(self, vocab_size=vocab_size, embed_dim=embedding_dim,
                             num_heads=4, num_layers=2, num_classes=num_classes, max_seq_len=512):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                    self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
                        dropout=0.1, batch_first=True
                    )
                    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                    self.fc = nn.Linear(embed_dim, num_classes)
                    self.max_seq_len = max_seq_len
                
                def forward(self, x):
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    seq_len = x.size(1)
                    embedded = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
                    output = self.transformer_encoder(embedded)
                    # Use mean pooling over sequence
                    pooled = output.mean(dim=1)
                    return self.fc(pooled)
            
            return TextTransformer()
        
        # Default to LSTM for text data
        return self._load_text_model('lstm', metadata)
    
    def _load_audio_model(self, architecture: str, metadata: ModelMetadata) -> nn.Module:
        """Load audio models (CNN, CRNN, etc.)"""
        input_channels = metadata.input_shape[0] if len(metadata.input_shape) > 1 else 1
        num_classes = metadata.output_classes
        
        # CNN-based audio classifier (for spectrograms)
        if 'cnn' in architecture or 'spectrogram' in architecture:
            class AudioCNN(nn.Module):
                def __init__(self, in_channels=input_channels, num_classes=num_classes):
                    super().__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(in_channels, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((4, 4))
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(),
                        nn.Linear(128 * 4 * 4, 256),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(256, num_classes)
                    )
                
                def forward(self, x):
                    if x.dim() == 3:
                        x = x.unsqueeze(1)
                    features = self.features(x)
                    return self.classifier(features)
            
            return AudioCNN()
        
        # CRNN (Convolutional Recurrent Neural Network) for audio
        if 'crnn' in architecture or 'rnn' in architecture:
            class AudioCRNN(nn.Module):
                def __init__(self, in_channels=input_channels, num_classes=num_classes):
                    super().__init__()
                    self.cnn = nn.Sequential(
                        nn.Conv2d(in_channels, 32, 3, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    )
                    self.rnn = nn.GRU(64, 128, batch_first=True, bidirectional=True)
                    self.fc = nn.Linear(128 * 2, num_classes)
                
                def forward(self, x):
                    if x.dim() == 3:
                        x = x.unsqueeze(1)
                    cnn_out = self.cnn(x)
                    # Convert to sequence: (batch, channels, freq, time) -> (batch, time, channels*freq)
                    cnn_out = cnn_out.squeeze(-1).permute(0, 2, 1)
                    rnn_out, hidden = self.rnn(cnn_out)
                    hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
                    return self.fc(hidden_cat)
            
            return AudioCRNN()
        
        # Default to CNN for audio data
        return self._load_audio_model('cnn', metadata)
    
    def _create_preprocessor(self, metadata: ModelMetadata) -> Any:
        """Create domain-specific input preprocessor"""
        domain = metadata.domain
        
        if domain == ModelDomain.VISION:
            return self._create_vision_preprocessor(metadata)
        elif domain == ModelDomain.TABULAR:
            return self._create_tabular_preprocessor(metadata)
        elif domain == ModelDomain.TEXT:
            return self._create_text_preprocessor(metadata)
        elif domain == ModelDomain.AUDIO:
            return self._create_audio_preprocessor(metadata)
        else:
            # Default identity preprocessor
            return lambda x: x
    
    def _create_vision_preprocessor(self, metadata: ModelMetadata) -> Any:
        """Create vision-specific preprocessor (normalization, resizing)"""
        # Standard ImageNet normalization values
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        def preprocess(x):
            if isinstance(x, np.ndarray):
                # Ensure float32 and normalize to [0, 1]
                x = x.astype(np.float32)
                if x.max() > 1.0:
                    x = x / 255.0
                
                # Handle different input shapes
                if x.ndim == 2:
                    x = np.expand_dims(x, axis=-1)  # Add channel dim
                
                if x.ndim == 3:
                    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
                
                # Normalize with ImageNet stats if RGB
                if x.shape[0] == 3:
                    x = (x - mean[:, None, None]) / std[:, None, None]
                elif x.shape[0] == 1:
                    # Grayscale - use single channel normalization
                    x = (x - 0.5) / 0.5
                
                return torch.from_numpy(x).unsqueeze(0)  # Add batch dim
            
            elif isinstance(x, torch.Tensor):
                if x.max() > 1.0:
                    x = x / 255.0
                return x.unsqueeze(0) if x.dim() == 3 else x
            
            return x
        
        return preprocess
    
    def _create_tabular_preprocessor(self, metadata: ModelMetadata) -> Any:
        """Create tabular-specific preprocessor (scaling, encoding)"""
        def preprocess(x):
            if isinstance(x, dict):
                # Convert dict of features to array
                x = np.array(list(x.values()), dtype=np.float32)
            
            if isinstance(x, list):
                x = np.array(x, dtype=np.float32)
            
            if isinstance(x, np.ndarray):
                # Handle missing values
                x = np.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
                
                # Clip extreme values
                x = np.clip(x, -1e6, 1e6)
                
                # Convert to tensor
                x = torch.from_numpy(x).float()
                
                # Ensure 2D (batch, features)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                
                return x
            
            elif isinstance(x, torch.Tensor):
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return x.float()
            
            return x
        
        return preprocess
    
    def _create_text_preprocessor(self, metadata: ModelMetadata) -> Any:
        """Create text-specific preprocessor (tokenization, padding)"""
        vocab_size = metadata.input_shape[0] if metadata.input_shape else 10000
        max_seq_len = metadata.input_shape[1] if len(metadata.input_shape) > 1 else 512
        
        def preprocess(x):
            if isinstance(x, str):
                # Simple tokenization: convert to lowercase and split
                tokens = x.lower().split()
                # Convert to indices (simple hash-based for demo)
                token_ids = [hash(token) % (vocab_size - 1) + 1 for token in tokens]
                x = np.array(token_ids[:max_seq_len], dtype=np.int64)
            
            if isinstance(x, list):
                if all(isinstance(item, str) for item in x):
                    # List of strings - tokenize each
                    token_ids = []
                    for text in x:
                        tokens = text.lower().split()
                        ids = [hash(token) % (vocab_size - 1) + 1 for token in tokens]
                        token_ids.extend(ids)
                    x = np.array(token_ids[:max_seq_len], dtype=np.int64)
                else:
                    x = np.array(x, dtype=np.int64)
            
            if isinstance(x, np.ndarray):
                # Truncate or pad to max_seq_len
                if len(x) > max_seq_len:
                    x = x[:max_seq_len]
                elif len(x) < max_seq_len:
                    x = np.pad(x, (0, max_seq_len - len(x)), constant_values=0)
                
                return torch.from_numpy(x).unsqueeze(0)  # Add batch dim
            
            elif isinstance(x, torch.Tensor):
                if x.dim() == 1:
                    if len(x) > max_seq_len:
                        x = x[:max_seq_len]
                    elif len(x) < max_seq_len:
                        x = torch.cat([x, torch.zeros(max_seq_len - len(x), dtype=x.dtype)])
                    x = x.unsqueeze(0)
                return x.long()
            
            return x
        
        return preprocess
    
    def _create_audio_preprocessor(self, metadata: ModelMetadata) -> Any:
        """Create audio-specific preprocessor (resampling, normalization)"""
        target_sample_rate = 16000  # Target sample rate in Hz
        
        def preprocess(x):
            if isinstance(x, np.ndarray):
                # Normalize audio to [-1, 1]
                if x.max() > 1.0 or x.min() < -1.0:
                    x = x / np.max(np.abs(x))
                
                # Ensure float32
                x = x.astype(np.float32)
                
                # Handle different shapes
                if x.ndim == 1:
                    x = np.expand_dims(x, axis=0)  # Add channel dim
                
                # Convert to tensor
                x = torch.from_numpy(x).float()
                
                # Add batch dimension if needed
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                
                return x
            
            elif isinstance(x, torch.Tensor):
                if x.max() > 1.0 or x.min() < -1.0:
                    x = x / torch.max(torch.abs(x))
                if x.dim() == 1:
                    x = x.unsqueeze(0).unsqueeze(0)
                elif x.dim() == 2:
                    x = x.unsqueeze(0)
                return x.float()
            
            return x
        
        return preprocess
    
    def _create_postprocessor(self, metadata: ModelMetadata) -> Any:
        """Create output postprocessor"""
        # TODO: Implement domain-specific postprocessing
        return lambda x: x  # Identity for now
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry from file"""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry to file"""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2, default=str)

# Helper function for datetime
from datetime import datetime
