"""
main execution script for dengue forecasting pipeline.
orchestrates the entire workflow from data loading to forecasting.
"""

import warnings
from typing import List

from src.utils.config import Config
from src.core.data_loader import DataLoader
from src.core.feature_engineer import FeatureEngineer
from src.core.model_trainer import ModelTrainer
from src.core.forecaster import Forecaster
from src.utils.visualizer import Visualizer
from src.monitoring.model_monitor import ModelMonitor
from src.utils.model_manager import ModelManager

warnings.filterwarnings('ignore')


class DengueForecastingPipeline:
    """
    main pipeline class following open/closed principle.
    orchestrates all components without knowing implementation details.
    """
    
    def __init__(self, enable_monitoring: bool = True, show_plots: bool = False):
        """
        initialize all pipeline components.
        
        args:
            enable_monitoring: whether to enable mlflow monitoring
            show_plots: whether to display plots during training
        """
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.forecaster = Forecaster(self.feature_engineer)
        self.visualizer = Visualizer(show_plots=show_plots)
        self.monitor = ModelMonitor() if enable_monitoring else None
        self.model_manager = ModelManager()
        
        self.df = None
        self.feature_cols = None
        self.enable_monitoring = enable_monitoring
        self.show_plots = show_plots
    
    def load_and_prepare_data(self) -> None:
        """load and prepare data for modeling."""
        print("\n" + "=" * 80)
        print("Loading and preparing data...")
        print("=" * 80)
        
        self.df = self.data_loader.load_and_prepare_data()
        self.df, self.feature_cols = self.feature_engineer.create_features(self.df)
        
        print(f"\n Data loaded: {len(self.df)} quarters")
        print(f" Features created: {len(self.feature_cols)} features")
    
    def evaluate_normal_year(self, test_year: int = 2023) -> dict:
        """
        evaluate model performance on a normal year.
        
        args:
            test_year: year to use for testing
            
        returns:
            dictionary with results
        """
        print("\n" + "=" * 80)
        print(f"TEST CASE: Predicting {test_year} (Normal Year)")
        print(f"Training: 2010-{test_year-1}")
        print("=" * 80)
        
        train_years = list(range(2010, test_year))
        
        results, y_test, test_df, valid_features, X_train = \
            self.model_trainer.train_and_evaluate(
                self.df, self.feature_cols, train_years, test_year
            )
        
        # print results
        self.model_trainer.print_results_summary(results, test_year)
        
        # get best model
        best_name, best_res = self.model_trainer.get_best_model(results)
        
        # save baseline model
        self.model_manager.save_baseline_model(
            model=best_res['model'],
            model_name=best_name,
            features=valid_features,
            metrics={
                'r2': best_res['r2'],
                'mae': best_res['mae'],
                'rmse': best_res['rmse']
            },
            test_year=test_year,
            train_years=train_years
        )
        print(f"\n Baseline model saved: {best_name} (R² = {best_res['r2']:.3f})")
        
        # mlflow monitoring
        if self.enable_monitoring and self.monitor:
            # log all models
            for model_name, model_res in results.items():
                metrics = {
                    'r2': model_res['r2'],
                    'mae': model_res['mae'],
                    'rmse': model_res['rmse']
                }
                if 'val_mae' in model_res:
                    metrics['val_mae'] = model_res['val_mae']

                run_id = self.monitor.log_training_run(
                    model=model_res['model'],
                    model_name=model_name,
                    params=model_res.get(
                        'best_params',
                        Config.RANDOM_FOREST_PARAMS if model_name == 'RandomForest'
                        else Config.GRADIENT_BOOSTING_PARAMS if model_name == 'GradientBoosting'
                        else Config.ADABOOST_PARAMS if model_name == 'AdaBoost'
                        else Config.PARAM_GRIDS.get('XGBoost', {})
                    ),
                    metrics=metrics,
                    features=valid_features,
                    train_year_range=(2010, test_year - 1),
                    test_year=test_year
                )
            
            # set baseline with best model
            self.monitor.set_baseline(
                metrics={
                    'r2': best_res['r2'],
                    'mae': best_res['mae'],
                    'rmse': best_res['rmse']
                },
                model_name=best_name,
                test_year=test_year
            )
        
        # print predictions
        self.model_trainer.print_predictions(
            test_df, y_test, best_res['predictions']
        )
        
        # show feature importance
        importance = self.model_trainer.get_feature_importance(
            best_res['model'], valid_features, top_n=10
        )
        self.visualizer.print_feature_importance(importance)
        
        return {
            'results': results,
            'y_test': y_test,
            'test_df': test_df,
            'valid_features': valid_features,
            'X_train': X_train,
            'best_name': best_name,
            'best_res': best_res
        }
    
    def evaluate_post_outbreak_year(
        self,
        test_year: int = 2025,
        exclude_year: int = 2024
    ) -> dict:
        """
        evaluate model performance on post-outbreak year.
        
        args:
            test_year: year to use for testing
            exclude_year: year to exclude from training
            
        returns:
            dictionary with results
        """
        print("\n" + "=" * 80)
        print(f"TEST CASE: Predicting {test_year} (Post-Outbreak Year)")
        print(f"Training: 2010-{test_year-1} (excluding {exclude_year})")
        print("=" * 80)
        
        train_years = [y for y in range(2010, test_year) if y != exclude_year]
        
        results, y_test, test_df, valid_features, _ = \
            self.model_trainer.train_and_evaluate(
                self.df, self.feature_cols, train_years, test_year
            )
        
        # print results
        self.model_trainer.print_results_summary(results, test_year)
        
        # get best model
        best_name, best_res = self.model_trainer.get_best_model(results)
        
        # save model with timestamp for comparison (NOT baseline - that's already set!)
        self.model_manager.save_model_with_timestamp(
            model=best_res['model'],
            model_name=best_name,
            features=valid_features,
            metrics={
                'r2': best_res['r2'],
                'mae': best_res['mae'],
                'rmse': best_res['rmse']
            },
            test_year=test_year,
            train_years=train_years
        )
        print(f"\n Model saved for year {test_year}: {best_name} (R² = {best_res['r2']:.3f})")
        
        # mlflow monitoring and drift detection
        if self.enable_monitoring and self.monitor:
            # log all models
            for model_name, model_res in results.items():
                metrics = {
                    'r2': model_res['r2'],
                    'mae': model_res['mae'],
                    'rmse': model_res['rmse']
                }
                if 'val_mae' in model_res:
                    metrics['val_mae'] = model_res['val_mae']

                run_id = self.monitor.log_training_run(
                    model=model_res['model'],
                    model_name=model_name,
                    params=model_res.get(
                        'best_params',
                        Config.RANDOM_FOREST_PARAMS if model_name == 'RandomForest'
                        else Config.GRADIENT_BOOSTING_PARAMS if model_name == 'GradientBoosting'
                        else Config.ADABOOST_PARAMS if model_name == 'AdaBoost'
                        else Config.PARAM_GRIDS.get('XGBoost', {})
                    ),
                    metrics=metrics,
                    features=valid_features,
                    train_year_range=(2010, test_year - 1),
                    test_year=test_year,
                    artifacts={'excluded_years': [exclude_year]}
                )
            
            # drift detection
            drift_detected, drift_reasons = self.monitor.detect_performance_drift(
                current_metrics={
                    'r2': best_res['r2'],
                    'mae': best_res['mae'],
                    'rmse': best_res['rmse']
                },
                model_name=best_name,
                test_year=test_year
            )
            
            # generate and print drift report
            report = self.monitor.generate_drift_report(
                current_metrics={
                    'r2': best_res['r2'],
                    'mae': best_res['mae'],
                    'rmse': best_res['rmse']
                },
                model_name=best_name,
                test_year=test_year,
                save_path=f"drift_report_{test_year}.txt"
            )
            print(report)
            
            # log drift alert
            self.monitor.log_drift_alert(
                drift_detected=drift_detected,
                drift_reasons=drift_reasons,
                model_name=best_name,
                test_year=test_year
            )
        
        # print predictions
        self.model_trainer.print_predictions(
            test_df, y_test, best_res['predictions']
        )
        
        return {
            'results': results,
            'y_test': y_test,
            'test_df': test_df,
            'best_name': best_name,
            'best_res': best_res
        }
    
    def generate_forecast(
        self,
        forecast_year: int,
        base_model,
        train_max_year: int,
        exclude_years: List[int] = None
    ) -> dict:
        """
        generate recursive forecast for future year.
        
        args:
            forecast_year: year to forecast
            base_model: base model to use for forecasting
            train_max_year: maximum year for training
            exclude_years: optional years to exclude
            
        returns:
            dictionary with forecast results
        """
        print("\n" + "=" * 80)
        print(f"Generating {forecast_year} Forecast")
        print(f"Training: 2010-{train_max_year}")
        if exclude_years:
            print(f"Excluding: {', '.join(map(str, exclude_years))}")
        print("=" * 80)
        
        forecast, fitted_model, valid_features = \
            self.forecaster.refit_and_forecast(
                self.df,
                self.feature_cols,
                base_model,
                forecast_year,
                train_max_year,
                exclude_years
            )
        
        # print forecast
        self.forecaster.print_forecast(forecast, forecast_year)
        
        return {
            'forecast': forecast,
            'fitted_model': fitted_model,
            'valid_features': valid_features
        }
    
    def plot_results(
        self,
        results_2023: dict,
        results_2025: dict,
        forecast_2026: dict
    ) -> None:
        """
        create visualizations for all results.
        
        args:
            results_2023: results from 2023 evaluation
            results_2025: results from 2025 evaluation
            forecast_2026: forecast for 2026
        """
        print("\n" + "=" * 80)
        print("Creating visualizations...")
        print("=" * 80)
        
        # plot 2023 results
        self.visualizer.plot_actual_vs_predicted(
            results_2023['test_df'],
            results_2023['best_res']['predictions'],
            results_2023['best_name'],
            2023,
            self.df,
            hist_years=2
        )
        
        # plot 2025 results
        self.visualizer.plot_actual_vs_predicted(
            results_2025['test_df'],
            results_2025['best_res']['predictions'],
            results_2025['best_name'],
            2025,
            self.df,
            hist_years=3
        )
        
        # plot 2026 forecast
        self.visualizer.plot_forecast(
            forecast_2026['forecast'],
            self.df,
            2026,
            'RandomForest',
            hist_start_year=2022
        )
    
    def print_summary(
        self,
        results_2023: dict,
        results_2025: dict
    ) -> None:
        """
        print final summary of all results.
        
        args:
            results_2023: results from 2023 evaluation
            results_2025: results from 2025 evaluation
        """
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        
        best_2023_name = results_2023['best_name']
        best_2023_r2 = results_2023['best_res']['r2']
        best_2023_mae = results_2023['best_res']['mae']
        
        best_2025_name = results_2025['best_name']
        best_2025_r2 = results_2025['best_res']['r2']
        best_2025_mae = results_2025['best_res']['mae']
        
        summary = f"""

                    HONEST MODEL PERFORMANCE (NO LEAKAGE)                    

                                                                             
  NORMAL YEAR (2023):                                                        
   Best Model: {best_2023_name:<20}                                       
   R² Score: {best_2023_r2:.4f}                                           
   MAE: {best_2023_mae:,.0f} cases                                        
                                                                             
  POST-OUTBREAK YEAR (2025):                                                 
   Best Model: {best_2025_name:<20}                                       
   R² Score: {best_2025_r2:.4f}  [WARNING]  (negative = worse than mean)        
   MAE: {best_2025_mae:,.0f} cases                                        
                                                                             
  KEY INSIGHTS:                                                              
  • Model performs well on normal years (R² = {best_2023_r2:.2f})            
  • Post-outbreak predictions challenging due to shifted baseline            
  • All features use only past data (no leakage)                             
  • Recursive forecasting enables multi-step ahead predictions               
                                                                             

"""
        print(summary)
        
        print("=" * 80)
        print("[OK] ANALYSIS COMPLETE")
        print("=" * 80)
    
    def run(self) -> None:
        """execute the complete pipeline."""
        print("\n" + "=" * 80)
        print("DENGUE FORECASTING - HONEST MODEL (NO DATA LEAKAGE)")
        if self.enable_monitoring:
            print("WITH MLFLOW MONITORING & DRIFT DETECTION")
        print("=" * 80)
        
        # load data
        self.load_and_prepare_data()
        
        # evaluate 2023 (normal year)
        results_2023 = self.evaluate_normal_year(test_year=2023)
        
        # evaluate 2025 (post-outbreak year)
        results_2025 = self.evaluate_post_outbreak_year(
            test_year=2025,
            exclude_year=2024
        )
        
        # generate 2026 forecast
        forecast_2026 = self.generate_forecast(
            forecast_year=2026,
            base_model=results_2023['best_res']['model'],
            train_max_year=2025,
            exclude_years=[2024]
        )
        
        # create visualizations
        self.plot_results(results_2023, results_2025, forecast_2026)
        
        # print summary
        self.print_summary(results_2023, results_2025)
        
        # export monitoring summary
        if self.enable_monitoring and self.monitor:
            self.monitor.export_monitoring_summary()
            
            # print metrics history
            history_df = self.monitor.get_metrics_history()
            if not history_df.empty:
                print("\n" + "=" * 80)
                print("[METRICS] METRICS HISTORY")
                print("=" * 80)
                print(history_df.to_string(index=False))


def main():
    """main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dengue Forecasting Pipeline")
    parser.add_argument(
        '--start-scheduler',
        action='store_true',
        help='Start drift scheduler after training completes'
    )
    parser.add_argument(
        '--scheduler-interval',
        type=str,
        default='daily',
        help='Drift check interval: daily, hourly, 6h, 30m, etc.'
    )
    args = parser.parse_args()
    
    # Run training pipeline
    pipeline = DengueForecastingPipeline(
        enable_monitoring=True,
        show_plots=False  # Disable plots during training
    )
    pipeline.run()
    
    # Optionally start drift scheduler
    if args.start_scheduler:
        print("\n" + "=" * 80)
        print("STARTING DRIFT SCHEDULER")
        print("=" * 80)
        print(f"Interval: {args.scheduler_interval}")
        print("Press Ctrl+C to stop both pipeline and scheduler\n")
        
        import subprocess
        import sys
        
        # Start scheduler in same process
        scheduler_script = 'scripts/drift_scheduler.py'
        subprocess.run([
            sys.executable, 
            scheduler_script,
            '--interval', args.scheduler_interval
        ])


if __name__ == '__main__':
    main()
