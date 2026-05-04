"""
🏗️ ENTERPRISE MODEL ROUTER - Multi-domain model management
Supported simultaneously: vision, tabular, text classifiers
"""
import json
import yaml
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
        
        if domain == ModelDomain.VISION:
            # Try to import from existing models
            try:
                from models.base.mnist_cnn import MNISTCNN
                return MNISTCNN()
            except ImportError:
                # Fallback to simple CNN
                class SimpleCNN(nn.Module):
                    def __init__(self, num_classes=metadata.output_classes):
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
                
                return SimpleCNN()
                
        elif domain == ModelDomain.TEXT:
            # Text classification model (simple LSTM-based)
            class TextClassifier(nn.Module):
                def __init__(self, vocab_size=10000, embed_dim=128, hidden_dim=256, num_classes=metadata.output_classes):
                    super().__init__()
                    self.embedding = nn.Embedding(vocab_size, embed_dim)
                    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
                    self.fc = nn.Linear(hidden_dim, num_classes)
                
                def forward(self, x):
                    # x shape: (batch, seq_len)
                    embedded = self.embedding(x)
                    _, (hidden, _) = self.lstm(embedded)
                    return self.fc(hidden[-1])
            
            return TextClassifier()
            
        elif domain == ModelDomain.AUDIO:
            # Audio classification model (simple CNN for spectrograms)
            class AudioClassifier(nn.Module):
                def __init__(self, num_classes=metadata.output_classes):
                    super().__init__()
                    # Assume input is a spectrogram-like 2D representation
                    self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                    self.pool = nn.MaxPool2d(2, 2)
                    self.fc1 = nn.Linear(64 * 32 * 32, 256)  # Adjust based on expected input
                    self.fc2 = nn.Linear(256, num_classes)
                
                def forward(self, x):
                    # x shape: (batch, 1, freq, time) or similar
                    if x.dim() == 2:
                        x = x.unsqueeze(1)  # Add channel dimension
                    x = self.pool(nn.functional.relu(self.conv1(x)))
                    x = self.pool(nn.functional.relu(self.conv2(x)))
                    x = torch.flatten(x, 1)
                    # Handle variable input sizes
                    if x.shape[1] != self.fc1.in_features:
                        # Adaptive pooling to match expected size
                        x = nn.functional.adaptive_avg_pool1d(x, self.fc1.in_features)
                    x = nn.functional.relu(self.fc1(x))
                    return self.fc2(x)
            
            return AudioClassifier()
            
        elif domain == ModelDomain.TABULAR:
            # Tabular data model (MLP)
            class TabularClassifier(nn.Module):
                def __init__(self, input_dim=100, hidden_dims=[128, 64], num_classes=metadata.output_classes):
                    super().__init__()
                    layers = []
                    prev_dim = input_dim
                    for hidden_dim in hidden_dims:
                        layers.append(nn.Linear(prev_dim, hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.BatchNorm1d(hidden_dim))
                        layers.append(nn.Dropout(0.3))
                        prev_dim = hidden_dim
                    layers.append(nn.Linear(prev_dim, num_classes))
                    self.network = nn.Sequential(*layers)
                
                def forward(self, x):
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    return self.network(x)
            
            return TabularClassifier()
            
        elif domain == ModelDomain.MULTIMODAL:
            # Multimodal model (placeholder - combines multiple encoders)
            class MultimodalClassifier(nn.Module):
                def __init__(self, num_classes=metadata.output_classes):
                    super().__init__()
                    # Simple fusion of different modalities
                    self.text_encoder = nn.LSTM(128, 256, batch_first=True)
                    self.image_encoder = nn.Sequential(
                        nn.Conv2d(3, 64, 3),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten()
                    )
                    self.fusion = nn.Linear(256 + 64, num_classes)
                
                def forward(self, x):
                    # Expecting dict with 'text' and 'image' keys
                    if isinstance(x, dict):
                        text_out = self.text_encoder(x.get('text', torch.zeros(1, 1, 128)))[0][:, -1, :]
                        image_out = self.image_encoder(x.get('image', torch.zeros(1, 3, 32, 32)))
                        combined = torch.cat([text_out, image_out], dim=1)
                    else:
                        # Fallback for simple input
                        combined = x
                    return self.fusion(combined)
            
            return MultimodalClassifier()
        else:
            raise NotImplementedError(f"Model loading not implemented for domain: {domain}")
    
    def _create_preprocessor(self, metadata: ModelMetadata) -> Any:
        """Create domain-specific input preprocessor"""
        domain = metadata.domain
        
        if domain == ModelDomain.VISION:
            # Vision preprocessing: normalize pixel values, resize, etc.
            import torch
            from torchvision import transforms
            
            # Standard image normalization for ImageNet-pretrained models
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
        elif domain == ModelDomain.TEXT:
            # Text preprocessing: tokenization, padding, truncation
            class TextPreprocessor:
                def __init__(self, max_length=512, vocab_size=10000):
                    self.max_length = max_length
                    self.vocab_size = vocab_size
                
                def __call__(self, text):
                    import torch
                    if isinstance(text, str):
                        text = [text]
                    
                    # Simple character-level tokenization (replace with BERT tokenizer in production)
                    tokens = []
                    for t in text:
                        # Convert to lowercase and take first max_length chars
                        t = t.lower()[:self.max_length]
                        # Simple ord-based tokenization
                        token_ids = [ord(c) % self.vocab_size for c in t]
                        # Pad to max_length
                        token_ids += [0] * (self.max_length - len(token_ids))
                        tokens.append(token_ids)
                    
                    return torch.tensor(tokens, dtype=torch.long)
            
            return TextPreprocessor()
            
        elif domain == ModelDomain.AUDIO:
            # Audio preprocessing: normalization, resampling, spectrogram
            import torch
            import numpy as np
            
            class AudioPreprocessor:
                def __init__(self, sample_rate=16000, n_mels=128, hop_length=512):
                    self.sample_rate = sample_rate
                    self.n_mels = n_mels
                    self.hop_length = hop_length
                
                def __call__(self, audio):
                    # Convert to numpy if tensor
                    if hasattr(audio, 'numpy'):
                        audio = audio.numpy()
                    
                    # Normalize audio to [-1, 1]
                    if isinstance(audio, np.ndarray):
                        audio = audio.astype(np.float32)
                        if np.abs(audio).max() > 0:
                            audio = audio / np.abs(audio).max()
                    
                    # Create spectrogram-like representation
                    # In production, use librosa.stft or torchaudio.transforms.MelSpectrogram
                    try:
                        import librosa
                        spec = librosa.feature.melspectrogram(
                            y=audio.flatten(),
                            sr=self.sample_rate,
                            n_mels=self.n_mels,
                            hop_length=self.hop_length
                        )
                        # Convert to log scale
                        spec = librosa.power_to_db(spec, ref=np.max)
                        # Normalize
                        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
                        return torch.from_numpy(spec).unsqueeze(0).float()
                    except ImportError:
                        # Fallback: simple FFT-based spectrogram
                        audio_flat = audio.flatten()
                        fft_result = np.fft.fft(audio_flat)
                        spec = np.abs(fft_result[:self.n_mels * self.hop_length])
                        spec = spec.reshape(self.n_mels, self.hop_length)
                        spec = np.log1p(spec)
                        spec = (spec - spec.mean()) / (spec.std() + 1e-8)
                        return torch.from_numpy(spec).unsqueeze(0).float()
            
            return AudioPreprocessor()
            
        elif domain == ModelDomain.TABULAR:
            # Tabular preprocessing: standardization, handling missing values
            import numpy as np
            import torch
            
            class TabularPreprocessor:
                def __init__(self, mean=None, std=None):
                    self.mean = mean
                    self.std = std
                
                def __call__(self, features):
                    # Convert to numpy array
                    if hasattr(features, 'numpy'):
                        features = features.numpy()
                    elif isinstance(features, list):
                        features = np.array(features, dtype=np.float32)
                    
                    # Handle missing values
                    features = np.nan_to_num(features, nan=0.0)
                    
                    # Standardize if statistics available
                    if self.mean is not None and self.std is not None:
                        features = (features - self.mean) / (self.std + 1e-8)
                    
                    return torch.from_numpy(features).float()
            
            return TabularPreprocessor()
            
        elif domain == ModelDomain.MULTIMODAL:
            # Multimodal preprocessing: combine text and image preprocessors
            from torchvision import transforms
            import torch
            
            class MultimodalPreprocessor:
                def __init__(self):
                    self.text_processor = TextPreprocessor()
                    self.image_processor = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                
                def __call__(self, data):
                    result = {}
                    if isinstance(data, dict):
                        if 'text' in data:
                            result['text'] = self.text_processor(data['text'])
                        if 'image' in data:
                            result['image'] = self.image_processor(data['image'])
                    return result
            
            return MultimodalPreprocessor()
        else:
            # Default: identity transform
            return lambda x: x
    
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
