"""
forecasting module for recursive multi-step ahead predictions.
handles refitting models and generating future forecasts.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.base import clone
from .feature_engineer import FeatureEngineer


class Forecaster:
    """
    forecaster class following single responsibility principle.
    responsible only for generating future predictions.
    """
    
    def __init__(self, feature_engineer: FeatureEngineer):
        """
        initialize forecaster with feature engineer.
        
        args:
            feature_engineer: feature engineer instance for creating features
        """
        self.feature_engineer = feature_engineer
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        max_year: int,
        exclude_years: List[int] = None
    ) -> pd.DataFrame:
        """
        prepare training data up to max_year.
        
        args:
            df: full dataframe
            max_year: maximum year to include in training
            exclude_years: optional list of years to exclude
            
        returns:
            filtered training dataframe
        """
        train_df = df[df['year'] <= max_year].copy()
        
        if exclude_years:
            train_df = train_df[~train_df['year'].isin(exclude_years)].copy()
        
        return train_df
    
    def fit_model(
        self,
        model: Any,
        train_df: pd.DataFrame,
        feature_cols: List[str]
    ) -> Tuple[Any, pd.DataFrame, List[str]]:
        """
        fit model on training data with feature engineering.
        
        args:
            model: sklearn model instance to fit
            train_df: training dataframe
            feature_cols: list of feature column names
            
        returns:
            tuple of (fitted model, X_train, valid_features)
        """
        # create features
        train_df_feat, _ = self.feature_engineer.create_features(train_df)
        
        # keep only rows with target
        train_df_feat = train_df_feat.dropna(subset=['casos_est']).copy()
        
        # get valid features
        valid_features = [c for c in feature_cols if c in train_df_feat.columns]
        
        # prepare training matrices
        X_train = train_df_feat[valid_features].copy()
        y_train = train_df_feat['casos_est'].values
        
        # remove all-nan columns
        all_nan_cols = [c for c in valid_features if X_train[c].isna().all()]
        valid_features = [c for c in valid_features if c not in all_nan_cols]
        X_train = train_df_feat[valid_features].copy()
        
        # drop rows with nans
        mask = ~X_train.isna().any(axis=1)
        X_train = X_train.loc[mask]
        y_train = y_train[mask.values]
        
        # clone and fit model
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)
        
        return fitted_model, X_train, valid_features
    
    def create_future_quarters(
        self,
        df: pd.DataFrame,
        forecast_year: int,
        num_quarters: int = 4
    ) -> pd.DataFrame:
        """
        create placeholder rows for future quarters.
        
        args:
            df: historical dataframe
            forecast_year: year to forecast
            num_quarters: number of quarters to forecast (default 4)
            
        returns:
            dataframe with future quarter placeholders
        """
        future_rows = []
        
        for q in range(1, num_quarters + 1):
            row = {
                'year': forecast_year,
                'quarter': q,
                'year_quarter': f'{forecast_year}-Q{q}',
                'casos_est': np.nan,
                'nino12': np.nan,
                'nino3': np.nan,
                'nino34': np.nan,
                'nino34_anom': np.nan,
            }
            future_rows.append(row)
        
        future_df = pd.DataFrame(future_rows)
        
        # fill enso values with persistence (last observed)
        last_obs = df[df['year'] <= forecast_year - 1].sort_values(
            ['year', 'quarter']
        ).iloc[-1]
        
        for col in ['nino12', 'nino3', 'nino34', 'nino34_anom']:
            if col in df.columns and pd.notna(last_obs.get(col, np.nan)):
                future_df[col] = last_obs[col]
        
        return future_df
    
    def recursive_forecast(
        self,
        model: Any,
        df: pd.DataFrame,
        forecast_year: int,
        valid_features: List[str],
        train_medians: pd.Series,
        num_quarters: int = 4
    ) -> pd.DataFrame:
        """
        perform recursive multi-step ahead forecasting.
        
        args:
            model: fitted model
            df: extended dataframe including future placeholders
            forecast_year: year being forecast
            valid_features: list of valid feature names
            train_medians: median values from training for imputation
            num_quarters: number of quarters to forecast
            
        returns:
            dataframe with forecasted values
        """
        extended = df.copy()
        predictions = []
        
        for step in range(num_quarters):
            quarter = step + 1
            year_quarter = f'{forecast_year}-Q{quarter}'
            
            # recompute features to include previous predictions
            ext_feat, _ = self.feature_engineer.create_features(extended)
            
            # find row for current quarter
            row_idx = ext_feat.index[ext_feat['year_quarter'] == year_quarter][0]
            X_row = ext_feat.loc[[row_idx], valid_features].copy()
            
            # fill missing values with training medians
            X_row = X_row.fillna(train_medians)
            
            # make prediction
            y_pred = float(model.predict(X_row)[0])
            predictions.append({
                'year_quarter': year_quarter,
                'predicted_casos_est': y_pred
            })
            
            # update extended dataframe with prediction for next iteration
            extended.loc[
                extended['year_quarter'] == year_quarter, 'casos_est'
            ] = y_pred
        
        return pd.DataFrame(predictions)
    
    def refit_and_forecast(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        base_model: Any,
        forecast_year: int,
        train_max_year: int,
        exclude_years: List[int] = None
    ) -> Tuple[pd.DataFrame, Any, List[str]]:
        """
        main method to refit model and generate forecast.
        
        args:
            df: full historical dataframe
            feature_cols: list of feature column names
            base_model: base model to clone and refit
            forecast_year: year to forecast
            train_max_year: maximum year to include in training
            exclude_years: optional years to exclude from training
            
        returns:
            tuple of (forecast dataframe, fitted model, valid features)
        """
        # prepare training data
        train_df = self.prepare_training_data(
            df, train_max_year, exclude_years
        )
        
        # fit model
        fitted_model, X_train, valid_features = self.fit_model(
            base_model, train_df, feature_cols
        )
        
        # create future quarters
        future_df = self.create_future_quarters(df, forecast_year)
        
        # combine historical and future data
        extended = pd.concat([df.copy(), future_df], ignore_index=True)
        extended = extended.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        # compute training medians for imputation
        train_medians = X_train.median(numeric_only=True)
        
        # perform recursive forecast
        forecast = self.recursive_forecast(
            fitted_model, extended, forecast_year,
            valid_features, train_medians
        )
        
        return forecast, fitted_model, valid_features

    def forecast_with_fitted_model(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fitted_model: Any,
        forecast_year: int,
        train_max_year: int,
        exclude_years: List[int] = None,
        feature_subset: List[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        generate forecast using an already-fitted model (no retraining).
        
        args:
            df: full historical dataframe
            feature_cols: list of feature column names
            fitted_model: pre-trained model
            forecast_year: year to forecast
            train_max_year: maximum year to include for stats
            exclude_years: optional years to exclude
            feature_subset: optional fixed feature list (e.g., from metadata)
        
        returns:
            tuple of (forecast dataframe, valid features)
        """
        # prepare data for stats/medians
        train_df = self.prepare_training_data(
            df, train_max_year, exclude_years
        )
        train_df_feat, _ = self.feature_engineer.create_features(train_df)
        train_df_feat = train_df_feat.dropna(subset=['casos_est']).copy()
        
        base_features = feature_subset or feature_cols
        valid_features = [c for c in base_features if c in train_df_feat.columns]
        
        X_train = train_df_feat[valid_features].copy()
        all_nan_cols = [c for c in valid_features if X_train[c].isna().all()]
        valid_features = [c for c in valid_features if c not in all_nan_cols]
        X_train = train_df_feat[valid_features].copy()
        
        # drop rows with nans for median computation
        mask = ~X_train.isna().any(axis=1)
        X_train = X_train.loc[mask]
        
        # create future quarters
        future_df = self.create_future_quarters(df, forecast_year)
        
        # combine historical and future data
        extended = pd.concat([df.copy(), future_df], ignore_index=True)
        extended = extended.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        # compute training medians for imputation
        train_medians = X_train.median(numeric_only=True)
        
        # perform recursive forecast
        forecast = self.recursive_forecast(
            fitted_model, extended, forecast_year,
            valid_features, train_medians
        )
        
        return forecast, valid_features
    
    def print_forecast(self, forecast: pd.DataFrame, year: int) -> None:
        """
        print formatted forecast table.
        
        args:
            forecast: dataframe with forecast results
            year: forecast year
        """
        print(f"\n {year} Quarterly Forecast:")
        print(f"{'Quarter':<15} {'Predicted Cases':>20}")
        print("-" * 40)
        
        for _, row in forecast.iterrows():
            print(
                f"{row['year_quarter']:<15} "
                f"{row['predicted_casos_est']:>20,.0f}"
            )
