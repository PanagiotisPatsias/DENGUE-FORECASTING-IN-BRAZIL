"""
data loading module for dengue forecasting.
handles loading and merging dengue and SST datasets.
"""

import pandas as pd
from typing import Tuple
from ..utils.config import Config


class DataLoader:
    """
    data loader class following single responsibility principle.
    responsible only for loading and preparing raw data.
    """
    
    def __init__(self, dengue_path: str = None, sst_path: str = None):
        """
        initialize data loader with file paths.
        
        args:
            dengue_path: path to dengue data csv file
            sst_path: path to sst indices csv file
        """
        self.dengue_path = dengue_path or Config.DENGUE_DATA_PATH
        self.sst_path = sst_path or Config.SST_DATA_PATH
    
    def load_dengue_data(self) -> pd.DataFrame:
        """
        load and process dengue dataset.
        
        returns:
            processed dengue dataframe with date and quarter columns
        """
        df = pd.read_csv(self.dengue_path)
        
        # parse dates
        df['data_iniSE'] = pd.to_datetime(df['data_iniSE'])
        df['year'] = df['data_iniSE'].dt.year
        df['quarter'] = df['data_iniSE'].dt.quarter
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        
        return df
    
    def load_sst_data(self) -> pd.DataFrame:
        """
        load and process SST indices dataset.
        
        returns:
            processed sst dataframe with date and quarter columns
        """
        df = pd.read_csv(self.sst_path)
        
        # parse dates
        df['date'] = pd.to_datetime(
            df['YR'].astype(str) + '-' + df['MON'].astype(str) + '-01'
        )
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['YR']
        df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
        
        return df
    
    def aggregate_sst_quarterly(self, sst_df: pd.DataFrame) -> pd.DataFrame:
        """
        aggregate SST data to quarterly level.
        
        args:
            sst_df: raw sst dataframe
            
        returns:
            quarterly aggregated sst dataframe
        """
        sst_quarterly = sst_df.groupby('year_quarter').agg({
            'NINO1+2': 'mean',
            'NINO3': 'mean',
            'NINO3.4': 'mean',
            'ANOM.3': 'mean',
            'year': 'first',
            'quarter': 'first'
        }).reset_index()
        
        sst_quarterly.columns = [
            'year_quarter', 'nino12', 'nino3', 'nino34',
            'nino34_anom', 'year', 'quarter'
        ]
        
        return sst_quarterly
    
    def aggregate_dengue_quarterly(self, dengue_df: pd.DataFrame) -> pd.DataFrame:
        """
        aggregate dengue data to quarterly level.
        only aggregates target variable to prevent data leakage.
        
        args:
            dengue_df: raw dengue dataframe
            
        returns:
            quarterly aggregated dengue dataframe
        """
        quarterly = dengue_df.groupby(['year', 'quarter', 'year_quarter']).agg({
            'casos_est': 'sum',
        }).reset_index()
        
        return quarterly
    
    def merge_datasets(
        self,
        dengue_quarterly: pd.DataFrame,
        sst_quarterly: pd.DataFrame
    ) -> pd.DataFrame:
        """
        merge dengue and sst quarterly datasets.
        
        args:
            dengue_quarterly: quarterly dengue data
            sst_quarterly: quarterly sst data
            
        returns:
            merged dataframe sorted by year and quarter
        """
        df = dengue_quarterly.merge(
            sst_quarterly,
            on='year_quarter',
            how='left',
            suffixes=('', '_sst')
        )
        
        # handle duplicate columns from merge
        df['year'] = df['year'].fillna(df['year_sst'])
        df['quarter'] = df['quarter'].fillna(df['quarter_sst'])
        df = df.drop(columns=['year_sst', 'quarter_sst'], errors='ignore')
        
        # sort by time
        df = df.sort_values(['year', 'quarter']).reset_index(drop=True)
        
        return df
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        main method to load and prepare all data.
        orchestrates the entire data loading pipeline.
        
        returns:
            fully prepared dataframe ready for feature engineering
        """
        # load raw data
        dengue_df = self.load_dengue_data()
        sst_df = self.load_sst_data()
        
        # aggregate to quarterly level
        sst_quarterly = self.aggregate_sst_quarterly(sst_df)
        dengue_quarterly = self.aggregate_dengue_quarterly(dengue_df)
        
        # merge datasets
        df = self.merge_datasets(dengue_quarterly, sst_quarterly)
        
        return df
