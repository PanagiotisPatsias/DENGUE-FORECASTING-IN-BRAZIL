"""
dengue forecasting package.
modular implementation following SOLID principles.
"""

from .utils.config import Config
from .core.data_loader import DataLoader
from .core.feature_engineer import FeatureEngineer
from .core.model_trainer import ModelTrainer
from .core.forecaster import Forecaster
from .utils.visualizer import Visualizer
from .monitoring.model_monitor import ModelMonitor

__all__ = [
    'Config',
    'DataLoader',
    'FeatureEngineer',
    'ModelTrainer',
    'Forecaster',
    'Visualizer',
    'ModelMonitor',
]

__version__ = '1.0.0'
