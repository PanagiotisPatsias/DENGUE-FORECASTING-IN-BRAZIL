"""
model training and evaluation module for dengue forecasting.
handles model creation, training, and performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
XGBRegressor = None
_XGBOOST_AVAILABLE = False
from ..utils.config import Config


class ModelTrainer:
    """
    model trainer class following single responsibility principle.
    responsible only for training and evaluating models.
    """
    
    def __init__(self):
        """initialize model trainer with configured models."""
        self.models = self._initialize_models()
    
    def _initialize_models(self) -> Dict[str, Any]:
        """
        initialize all models with configured hyperparameters.
        
        returns:
            dictionary of model name to model instance
        """
        models = {
            'GradientBoosting': GradientBoostingRegressor(
                **Config.GRADIENT_BOOSTING_PARAMS
            ),
            'RandomForest': RandomForestRegressor(
                **Config.RANDOM_FOREST_PARAMS
            ),
            'AdaBoost': AdaBoostRegressor(
                **Config.ADABOOST_PARAMS
            ),
        }

        print("[INFO] XGBoost disabled. Skipping XGBoost model.")

        return models
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_years: List[int],
        test_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        prepare training and test datasets.
        
        args:
            df: full dataframe with features
            feature_cols: list of feature column names
            train_years: list of years to use for training
            test_year: year to use for testing
            
        returns:
            tuple of (train_df, test_df, valid_features)
        """
        train_df = df[df['year'].isin(train_years)].copy()
        test_df = df[df['year'] == test_year].copy()
        
        # filter features based on validity
        valid_features = self._get_valid_features(train_df, feature_cols)
        
        return train_df, test_df, valid_features
    
    def _get_valid_features(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> List[str]:
        """
        get valid features that have sufficient non-null values.
        
        args:
            train_df: training dataframe
            feature_cols: list of candidate feature columns
            
        returns:
            list of valid feature column names
        """
        min_ratio = Config.MIN_FEATURE_VALID_RATIO
        return [
            col for col in feature_cols
            if train_df[col].notna().sum() > len(train_df) * min_ratio
        ]
    
    def prepare_feature_matrices(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        valid_features: List[str]
    ) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """
        prepare feature matrices and target vectors.
        
        args:
            train_df: training dataframe
            test_df: test dataframe
            valid_features: list of valid feature columns
            
        returns:
            tuple of (X_train, y_train, X_test, y_test)
        """
        X_train = train_df[valid_features].fillna(
            train_df[valid_features].median()
        )
        y_train = train_df['casos_est'].values
        
        X_test = test_df[valid_features].fillna(
            train_df[valid_features].median()
        )
        y_test = test_df['casos_est'].values
        
        return X_train, y_train, X_test, y_test
    
    def train_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray
    ) -> Any:
        """
        train a single model.
        
        args:
            model: sklearn model instance
            X_train: training features
            y_train: training target
            
        returns:
            trained model
        """
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        evaluate trained model on test data.
        
        args:
            model: trained model
            X_test: test features
            y_test: test target
            
        returns:
            dictionary with evaluation metrics and predictions
        """
        y_pred = np.maximum(0, model.predict(X_test))
        
        return {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'predictions': y_pred,
            'model': model
        }
    
    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        train_years: List[int],
        test_year: int
    ) -> Tuple[Dict[str, Dict[str, Any]], np.ndarray, pd.DataFrame, List[str], pd.DataFrame]:
        """
        main method to train and evaluate all models.
        
        args:
            df: full dataframe with features
            feature_cols: list of feature column names
            train_years: list of years for training
            test_year: year for testing
            
        returns:
            tuple of (results dict, y_test, test_df, valid_features, X_train)
        """
        # prepare data
        train_df, test_df, valid_features = self.prepare_training_data(
            df, feature_cols, train_years, test_year
        )
        
        X_train, y_train, X_test, y_test = self.prepare_feature_matrices(
            train_df, test_df, valid_features
        )
        
        # train and evaluate all models
        results = {}
        for name, model in self.models.items():
            if Config.ENABLE_GRID_SEARCH and name in Config.PARAM_GRIDS:
                tscv = TimeSeriesSplit(n_splits=Config.TSCV_SPLITS)
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=Config.PARAM_GRIDS[name],
                    cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                evaluation = self.evaluate_model(best_model, X_test, y_test)
                evaluation['val_mae'] = -grid_search.best_score_
                evaluation['best_params'] = grid_search.best_params_
                results[name] = evaluation
            else:
                trained_model = self.train_model(model, X_train, y_train)
                results[name] = self.evaluate_model(trained_model, X_test, y_test)
        
        return results, y_test, test_df, valid_features, X_train
    
    def get_best_model(
        self,
        results: Dict[str, Dict[str, Any]]
    ) -> Tuple[str, Dict[str, Any]]:
        """
        get the best performing model based on r2 score.
        
        args:
            results: dictionary of model results
            
        returns:
            tuple of (best model name, best model results)
        """
        return max(results.items(), key=lambda x: x[1]['r2'])
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        get feature importance from trained model.
        
        args:
            model: trained model with feature_importances_ attribute
            feature_names: list of feature names
            top_n: number of top features to return
            
        returns:
            dataframe with features and their importance scores
        """
        if not hasattr(model, 'feature_importances_'):
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance.head(top_n)
    
    def print_results_summary(
        self,
        results: Dict[str, Dict[str, Any]],
        test_year: int,
        description: str = ""
    ) -> None:
        """
        print formatted summary of model results.
        
        args:
            results: dictionary of model results
            test_year: year being tested
            description: optional description of the test scenario
        """
        print("\n" + "=" * 80)
        print(f"Results for {test_year}")
        if description:
            print(description)
        print("=" * 80)
        
        print(f"\n{'Model':<20} {'R²':>10} {'MAE':>15} {'RMSE':>15}")
        print("-" * 60)
        
        for name, res in results.items():
            print(
                f"{name:<20} {res['r2']:>10.4f} "
                f"{res['mae']:>15,.0f} {res['rmse']:>15,.0f}"
            )
        
        best_name, best_res = self.get_best_model(results)
        print(f"\n Best: {best_name} with R² = {best_res['r2']:.4f}")
    
    def print_predictions(
        self,
        test_df: pd.DataFrame,
        y_test: np.ndarray,
        predictions: np.ndarray
    ) -> None:
        """
        print formatted prediction table.
        
        args:
            test_df: test dataframe with year_quarter column
            y_test: actual target values
            predictions: predicted values
        """
        print("\n[CHART] Predictions:")
        print(f"{'Quarter':<12} {'Actual':>15} {'Predicted':>15} {'Error':>15}")
        print("-" * 60)
        
        for i, (_, row) in enumerate(test_df.iterrows()):
            error = y_test[i] - predictions[i]
            print(
                f"{row['year_quarter']:<12} {y_test[i]:>15,.0f} "
                f"{predictions[i]:>15,.0f} {error:>15,.0f}"
            )
