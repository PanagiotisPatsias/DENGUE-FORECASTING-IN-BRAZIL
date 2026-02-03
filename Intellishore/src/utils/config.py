"""
configuration module for dengue forecasting models.
contains model parameters, file paths, and constants.
"""

import os
import copy
from typing import Dict, Any


class Config:
    """configuration class following single responsibility principle."""
    
    # file paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    DENGUE_DATA_PATH = os.path.join(DATA_DIR, 'infodengue_capitals_subsetBR.csv')
    SST_DATA_PATH = os.path.join(DATA_DIR, 'sst_indices.csv')
    
    # model hyperparameters
    GRADIENT_BOOSTING_PARAMS: Dict[str, Any] = {
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'random_state': 42
    }
    
    RANDOM_FOREST_PARAMS: Dict[str, Any] = {
        'n_estimators': 300,
        'max_depth': 6,
        'min_samples_leaf': 2,
        'random_state': 42
    }
    
    ADABOOST_PARAMS: Dict[str, Any] = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'random_state': 42
    }

    # grid search configuration
    ENABLE_GRID_SEARCH = True
    TSCV_SPLITS = 3
    PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
        'RandomForest': {
            'n_estimators': [100, 300],
            'max_depth': [4, 6],
            'min_samples_leaf': [2, 4],
            'random_state': [42]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.05],
            'max_depth': [3, 4],
            'random_state': [42]
        },
        'AdaBoost': {
            'n_estimators': [100],
            'learning_rate': [0.1],
            'random_state': [42]
        }
    }
    DEFAULT_ENABLE_GRID_SEARCH = ENABLE_GRID_SEARCH
    DEFAULT_TSCV_SPLITS = TSCV_SPLITS
    DEFAULT_PARAM_GRIDS = copy.deepcopy(PARAM_GRIDS)
    
    # feature engineering parameters
    LAG_QUARTERS = [1, 2, 3, 4, 5, 6, 7, 8]
    ROLLING_WINDOWS = [2, 4, 8]
    EMA_SPANS = [2, 4, 8]
    
    # training parameters
    MIN_FEATURE_VALID_RATIO = 0.5
    
    # columns to exclude from features
    EXCLUDED_COLUMNS = [
        'year', 'quarter', 'year_quarter', 'casos_est',
        'nino12', 'nino3', 'nino34', 'nino34_anom'
    ]
