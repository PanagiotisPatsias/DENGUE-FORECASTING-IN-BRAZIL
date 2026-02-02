"""
model persistence manager for saving and loading trained models.
handles serialization of models, metadata, and feature information.
"""

import os
import pickle
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .config import Config


class ModelManager:
    """
    model manager class for persistence operations.
    follows single responsibility principle for model I/O.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        """
        initialize model manager.
        
        args:
            models_dir: directory to store models (default: Config.MODELS_DIR)
        """
        self.models_dir = Path(models_dir) if models_dir else Path(Config.MODELS_DIR)
        self._ensure_models_directory()
    
    def _ensure_models_directory(self) -> None:
        """create models directory if it doesn't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_baseline_model(
        self,
        model: Any,
        model_name: str,
        features: List[str],
        metrics: Dict[str, float],
        test_year: int,
        train_years: List[int]
    ) -> str:
        """
        save the baseline model with metadata.
        
        args:
            model: trained model instance
            model_name: name of the model (e.g., 'RandomForest')
            features: list of feature names used
            metrics: evaluation metrics (r2, mae, rmse)
            test_year: year used for testing
            train_years: years used for training
            
        returns:
            path to saved model file
        """
        # create baseline model path
        model_filename = "baseline_model.pkl"
        model_path = self.models_dir / model_filename
        
        # create metadata
        metadata = {
            'model_name': model_name,
            'features': features,
            'metrics': metrics,
            'test_year': test_year,
            'train_years': train_years,
            'saved_at': datetime.now().isoformat(),
            'num_features': len(features)
        }
        
        # save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # save metadata
        metadata_path = self.models_dir / "baseline_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  → Model saved to: {model_path}")
        print(f"  → Metadata saved to: {metadata_path}")
        
        return str(model_path)
    
    def save_model_with_timestamp(
        self,
        model: Any,
        model_name: str,
        features: List[str],
        metrics: Dict[str, float],
        test_year: int,
        train_years: List[int]
    ) -> str:
        """
        save a model with timestamp (for versioning).
        
        args:
            model: trained model instance
            model_name: name of the model
            features: list of feature names
            metrics: evaluation metrics
            test_year: year used for testing
            train_years: years used for training
            
        returns:
            path to saved model file
        """
        # create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name}_test{test_year}_{timestamp}.pkl"
        model_path = self.models_dir / model_filename
        
        # create metadata
        metadata = {
            'model_name': model_name,
            'features': features,
            'metrics': metrics,
            'test_year': test_year,
            'train_years': train_years,
            'saved_at': datetime.now().isoformat(),
            'num_features': len(features)
        }
        
        # save model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # save metadata
        metadata_filename = f"{model_name}_test{test_year}_{timestamp}_metadata.json"
        metadata_path = self.models_dir / metadata_filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(model_path)
    
    def load_baseline_model(self) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        load the baseline model and its metadata.
        
        returns:
            tuple of (model, metadata) or None if not found
        """
        model_path = self.models_dir / "baseline_model.pkl"
        metadata_path = self.models_dir / "baseline_metadata.json"
        
        if not model_path.exists():
            print(f"[WARNING]  No baseline model found at {model_path}")
            return None
        
        # load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # load metadata
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        print(f" Loaded baseline model: {metadata.get('model_name', 'Unknown')}")
        print(f"  → Features: {metadata.get('num_features', 'Unknown')}")
        print(f"  → Baseline R²: {metadata.get('metrics', {}).get('r2', 'Unknown')}")
        print(f"  → Test Year: {metadata.get('test_year', 'Unknown')}")
        
        return model, metadata
    
    def load_model_by_path(self, model_path: str) -> Any:
        """
        load a model from a specific path.
        
        args:
            model_path: path to the model file
            
        returns:
            loaded model
        """
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def list_saved_models(self) -> List[Dict[str, str]]:
        """
        list all saved models in the models directory.
        
        returns:
            list of dictionaries with model information
        """
        models = []
        
        for model_file in self.models_dir.glob("*.pkl"):
            metadata_file = model_file.with_suffix('').with_name(
                model_file.stem + "_metadata.json"
            )
            
            model_info = {
                'filename': model_file.name,
                'path': str(model_file),
                'size': f"{model_file.stat().st_size / 1024:.2f} KB",
                'modified': datetime.fromtimestamp(
                    model_file.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # add metadata if available
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    model_info.update({
                        'model_name': metadata.get('model_name', 'Unknown'),
                        'r2': metadata.get('metrics', {}).get('r2', 'Unknown'),
                        'test_year': metadata.get('test_year', 'Unknown')
                    })
            
            models.append(model_info)
        
        return models
    
    def delete_model(self, model_filename: str) -> bool:
        """
        delete a saved model and its metadata.
        
        args:
            model_filename: name of the model file to delete
            
        returns:
            True if successful, False otherwise
        """
        model_path = self.models_dir / model_filename
        
        if not model_path.exists():
            print(f"[WARNING]  Model not found: {model_filename}")
            return False
        
        # delete model file
        model_path.unlink()
        
        # delete metadata if exists
        metadata_path = model_path.with_suffix('').with_name(
            model_path.stem + "_metadata.json"
        )
        if metadata_path.exists():
            metadata_path.unlink()
        
        print(f" Deleted model: {model_filename}")
        return True
    
    def get_baseline_metadata(self) -> Optional[Dict[str, Any]]:
        """
        get metadata for the baseline model without loading the model.
        
        returns:
            metadata dictionary or None if not found
        """
        metadata_path = self.models_dir / "baseline_metadata.json"
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
