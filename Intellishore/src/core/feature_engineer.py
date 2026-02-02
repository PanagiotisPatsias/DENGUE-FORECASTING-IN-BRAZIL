"""
feature engineering module for dengue forecasting.
creates lag features, rolling statistics, and seasonal features.
ensures no data leakage by using only past information.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from ..utils.config import Config


class FeatureEngineer:
    """
    feature engineer class following single responsibility principle.
    responsible only for creating features from prepared data.
    """
    
    def __init__(
        self,
        lag_quarters: List[int] = None,
        rolling_windows: List[int] = None,
        ema_spans: List[int] = None
    ):
        """
        initialize feature engineer with configuration.
        
        args:
            lag_quarters: list of lag periods for creating lag features
            rolling_windows: list of window sizes for rolling statistics
            ema_spans: list of span values for exponential moving averages
        """
        self.lag_quarters = lag_quarters or Config.LAG_QUARTERS
        self.rolling_windows = rolling_windows or Config.ROLLING_WINDOWS
        self.ema_spans = ema_spans or Config.EMA_SPANS
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create lag features for target variable.
        uses past quarters to prevent data leakage.
        
        args:
            df: input dataframe with casos_est column
            
        returns:
            dataframe with lag features added
        """
        df_copy = df.copy()
        
        for lag in self.lag_quarters:
            df_copy[f'lag_{lag}'] = df_copy['casos_est'].shift(lag)
        
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create rolling window statistics from past data only.
        uses shift(1) to ensure no current quarter leakage.
        
        args:
            df: input dataframe with casos_est column
            
        returns:
            dataframe with rolling features added
        """
        df_copy = df.copy()
        
        for window in self.rolling_windows:
            shifted = df_copy['casos_est'].shift(1)
            df_copy[f'roll_mean_{window}'] = shifted.rolling(
                window, min_periods=1
            ).mean()
            df_copy[f'roll_std_{window}'] = shifted.rolling(
                window, min_periods=1
            ).std()
            df_copy[f'roll_max_{window}'] = shifted.rolling(
                window, min_periods=1
            ).max()
            df_copy[f'roll_min_{window}'] = shifted.rolling(
                window, min_periods=1
            ).min()
        
        return df_copy
    
    def create_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create exponential moving average features.
        
        args:
            df: input dataframe with casos_est column
            
        returns:
            dataframe with ema features added
        """
        df_copy = df.copy()
        
        for span in self.ema_spans:
            df_copy[f'ema_{span}'] = df_copy['casos_est'].shift(1).ewm(
                span=span, adjust=False
            ).mean()
        
        return df_copy
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create seasonal features based on quarter.
        these are known at forecast time so no leakage.
        
        args:
            df: input dataframe with quarter column
            
        returns:
            dataframe with seasonal features added
        """
        df_copy = df.copy()
        
        # cyclical encoding of quarters
        df_copy['quarter_sin'] = np.sin(
            2 * np.pi * df_copy['quarter'].astype(int) / 4
        )
        df_copy['quarter_cos'] = np.cos(
            2 * np.pi * df_copy['quarter'].astype(int) / 4
        )
        
        # peak season indicator (Q1 and Q2)
        df_copy['is_peak_season'] = (
            (df_copy['quarter'] == 1) | (df_copy['quarter'] == 2)
        ).astype(int)
        
        return df_copy
    
    def create_year_over_year_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create year-over-year comparison features.
        
        args:
            df: input dataframe with casos_est column
            
        returns:
            dataframe with yoy features added
        """
        df_copy = df.copy()
        
        # same quarter previous year
        df_copy['yoy_same_quarter'] = df_copy['casos_est'].shift(4)
        
        # yoy growth rate
        df_copy['yoy_growth_rate'] = df_copy['casos_est'].shift(1) / (
            df_copy['casos_est'].shift(5) + 1
        )
        
        return df_copy
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create trend features based on year.
        
        args:
            df: input dataframe with year column
            
        returns:
            dataframe with trend features added
        """
        df_copy = df.copy()
        
        df_copy['year_trend'] = df_copy['year'] - df_copy['year'].min()
        
        return df_copy
    
    def create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create momentum and acceleration features.
        
        args:
            df: input dataframe with casos_est column
            
        returns:
            dataframe with momentum features added
        """
        df_copy = df.copy()
        
        # momentum (rate of change)
        df_copy['momentum_1q'] = (
            df_copy['casos_est'].shift(1) - df_copy['casos_est'].shift(2)
        )
        
        # acceleration (change in momentum)
        df_copy['acceleration'] = (
            df_copy['momentum_1q'] - df_copy['momentum_1q'].shift(1)
        )
        
        return df_copy
    
    def create_log_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create log-transformed features for better distribution.
        
        args:
            df: input dataframe
            
        returns:
            dataframe with log features added
        """
        df_copy = df.copy()
        
        df_copy['log_lag_1'] = np.log1p(df_copy['casos_est'].shift(1))
        df_copy['log_lag_4'] = np.log1p(df_copy['casos_est'].shift(4))
        df_copy['log_roll_mean_4'] = np.log1p(df_copy['roll_mean_4'])
        
        return df_copy
    
    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create ratio features comparing current to historical values.
        
        args:
            df: input dataframe
            
        returns:
            dataframe with ratio features added
        """
        df_copy = df.copy()
        
        df_copy['ratio_to_roll_mean'] = df_copy['casos_est'].shift(1) / (
            df_copy['roll_mean_4'] + 1
        )
        
        return df_copy
    
    def create_enso_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        create el niño features with appropriate lags.
        
        args:
            df: input dataframe with nino34 column
            
        returns:
            dataframe with enso features added
        """
        df_copy = df.copy()
        
        if 'nino34' in df_copy.columns:
            df_copy['nino34_lag1'] = df_copy['nino34'].shift(1)
        
        return df_copy
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        get list of feature columns excluding target and metadata.
        
        args:
            df: dataframe with all columns
            
        returns:
            list of feature column names
        """
        return [
            col for col in df.columns
            if col not in Config.EXCLUDED_COLUMNS
        ]
    
    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        main method to create all features.
        orchestrates the entire feature engineering pipeline.
        
        args:
            df: input dataframe with raw data
            
        returns:
            tuple of (dataframe with features, list of feature column names)
        """
        df_feat = df.copy()
        
        # create all feature groups
        df_feat = self.create_lag_features(df_feat)
        df_feat = self.create_rolling_features(df_feat)
        df_feat = self.create_ema_features(df_feat)
        df_feat = self.create_seasonal_features(df_feat)
        df_feat = self.create_year_over_year_features(df_feat)
        df_feat = self.create_trend_features(df_feat)
        df_feat = self.create_momentum_features(df_feat)
        df_feat = self.create_log_features(df_feat)
        df_feat = self.create_ratio_features(df_feat)
        df_feat = self.create_enso_features(df_feat)
        
        # get feature column names
        feature_cols = self.get_feature_columns(df_feat)
        
        return df_feat, feature_cols
