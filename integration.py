"""
Integration Module: Connecting Ensemble Sentiment Analyzer with Future Work
==============================================================================
This module bridges your existing FinLlama ensemble system with the new
time series analysis, portfolio optimization, and visualization features.

Usage:
    from integration import SentimentPipeline
    
    pipeline = SentimentPipeline()
    pipeline.run_full_analysis(companies=['AAPL', 'GOOGL', 'MSFT'])
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

# Import future work modules
from future_work_implementation import (
    TimeSeriesAnalyzer,
    PortfolioOptimizer,
    SentimentAlertSystem,
    EnhancedVisualizer
)


class SentimentPipeline:
    """
    End-to-end pipeline integrating ensemble sentiment analysis with
    advanced analytics and portfolio optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the pipeline
        
        Args:
            config: Configuration dictionary with parameters like:
                - lookback_days: Number of days to analyze (default: 60)
                - forecast_days: Days to forecast ahead (default: 7)
                - risk_free_rate: Risk-free rate for portfolio optimization (default: 0.02)
        """
        self.config = config or {}
        self.lookback_days = self.config.get('lookback_days', 60)
        self.forecast_days = self.config.get('forecast_days', 7)
        self.risk_free_rate = self.config.get('risk_free_rate', 0.02)
        
        # Initialize components
        self.ts_analyzer = TimeSeriesAnalyzer(sequence_length=10)
        self.portfolio_optimizer = PortfolioOptimizer(risk_free_rate=self.risk_free_rate)
        self.alert_system = SentimentAlertSystem(lookback_period=30, z_threshold=2.0)
        self.visualizer = EnhancedVisualizer()
        
        self.results = {}
        
    def load_ensemble_output(self, filepath: str) -> pd.DataFrame:
        """
        Load sentiment scores from your ensemble system output
        
        Expected CSV format:
            date, company, sentiment_score, confidence, textblob_score, 
            finbert_score, keyword_score, regime_score, summary_score, temporal_score
        
        Args:
            filepath: Path to CSV file with ensemble output
        
        Returns:
            DataFrame with sentiment data
        """
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def prepare_company_data(self, df: pd.DataFrame, company: str) -> pd.DataFrame:
        """
        Prepare data for a specific company
        
        Args:
            df: Full sentiment DataFrame
            company: Company symbol
        
        Returns:
            Filtered and sorted DataFrame
        """
        company_df = df[df['company'] == company].copy()
        company_df = company_df.sort_values('date').reset_index(drop=True)
        
        # Ensure required columns exist
        if 'volume' not in company_df.columns:
            company_df['volume'] = 100  # Default volume if not available
        
        return company_df
    
    def run_time_series_analysis(self, sentiment_data: Dict[str, pd.DataFrame], 
                                 train_models: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Run time series analysis and forecasting for all companies
        
        Args:
            sentiment_data: Dict mapping company symbols to sentiment DataFrames
            train_models: Whether to train new models (can be slow)
        
        Returns:
            Dictionary mapping company symbols to prediction DataFrames
        """
        predictions = {}
        
        for company, df in sentiment_data.items():
            print(f"Analyzing {company}...")
            
            if train_models and len(df) >= 30:  # Need minimum data for training
                try:
                    # Train model
                    history = self.ts_analyzer.train_lstm_model(
                        df, 
                        epochs=30, 
                        batch_size=16, 
                        learning_rate=0.001
                    )
                    
                    # Make predictions
                    pred_df = self.ts_analyzer.predict_future_sentiment(
                        df, 
                        days_ahead=self.forecast_days
                    )
                    predictions[company] = pred_df
                    
                    print(f"  ✓ Generated {len(pred_df)} day forecast")
                    
                except Exception as e:
                    print(f"  ✗ Error: {str(e)}")
                    predictions[company] = None
            else:
                # Use simple moving average forecast if not enough data
                last_sentiment = df['sentiment_score'].tail(7).mean()
                future_dates = [
                    df['date'].iloc[-1] + timedelta(days=i+1) 
                    for i in range(self.forecast_days)
                ]
                
                predictions[company] = pd.DataFrame({
                    'date': future_dates,
                    'predicted_sentiment': [last_sentiment] * self.forecast_days,
                    'prediction_type': 'simple_average'
                })
        
        self.results['predictions'] = predictions
        return predictions
    
    def run_portfolio_optimization(self, sentiment_data: Dict[str, pd.DataFrame],
                                  returns_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Optimize portfolio allocation based on sentiment signals
        
        Args:
            sentiment_data: Dict mapping company symbols to sentiment DataFrames
            returns_data: Historical returns (if None, generates synthetic data)
        
        Returns:
            Dictionary with optimization results
        """
        companies = list(sentiment_data.keys())
        
        # Get latest sentiment scores
        sentiment_scores = {
            company: df['sentiment_score'].iloc[-1]
            for company, df in sentiment_data.items()
        }
        
        # Generate or use provided returns data
        if returns_data is None:
            print("Generating synthetic returns data for demonstration...")
            dates = pd.date_range(
                start=min([df['date'].min() for df in sentiment_data.values()]),
                end=max([df['date'].max() for df in sentiment_data.values()]),
                freq='D'
            )
            returns_data = pd.DataFrame({
                company: np.random.randn(len(dates)) * 0.02 + 0.001
                for company in companies
            })
        
        # Calculate sentiment-weighted expected returns
        expected_returns = self.portfolio_optimizer.calculate_sentiment_weighted_returns(
            sentiment_scores, returns_data
        )
        
        # Calculate covariance matrix
        cov_matrix = returns_data.cov().values
        
        # Optimize for maximum Sharpe ratio
        optimal_sharpe = self.portfolio_optimizer.optimize_portfolio_sharpe(
            expected_returns, cov_matrix
        )
        
        # Optimize for minimum variance
        optimal_minvar = self.portfolio_optimizer.optimize_portfolio_min_variance(
            cov_matrix
        )
        
        # Generate efficient frontier
        frontier_df = self.portfolio_optimizer.generate_efficient_frontier(
            expected_returns, cov_matrix, n_points=50
        )
        
        portfolio_results = {
            'companies': companies,
            'sentiment_scores': sentiment_scores,
            'expected_returns': expected_returns,
            'optimal_sharpe': optimal_sharpe,
            'optimal_minvar': optimal_minvar,
            'efficient_frontier': frontier_df
        }
        
        self.results['portfolio'] = portfolio_results
        
        print("\nOptimal Portfolio (Maximum Sharpe Ratio):")
        print("-" * 60)
        for i, company in enumerate(companies):
            weight = optimal_sharpe['weights'][i]
            if weight > 0.01:  # Only show significant allocations
                print(f"  {company:6s}: {weight:6.2%}  (Sentiment: {sentiment_scores[company]:+.3f})")
        print(f"\nExpected Return: {optimal_sharpe['expected_return']:.4f}")
        print(f"Volatility:      {optimal_sharpe['volatility']:.4f}")
        print(f"Sharpe Ratio:    {optimal_sharpe['sharpe_ratio']:.4f}")
        
        return portfolio_results
    
    def generate_alerts(self, sentiment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate alerts for sentiment anomalies
        
        Args:
            sentiment_data: Dict mapping company symbols to sentiment DataFrames
        
        Returns:
            DataFrame with all alerts
        """
        alerts_df = self.alert_system.get_all_alerts(sentiment_data)
        
        if len(alerts_df) > 0:
            print(f"\n⚠️  Generated {len(alerts_df)} alerts:")
            print("-" * 60)
            
            for _, alert in alerts_df.iterrows():
                severity_icon = "🔴" if alert.get('severity') == 'high' else "🟡"
                alert_type = alert.get('alert_type', alert.get('reversal_type', 'unknown'))
                company = alert.get('company', 'N/A')
                
                print(f"{severity_icon} {company}: {alert_type}")
        else:
            print("\n✓ No alerts detected")
        
        self.results['alerts'] = alerts_df
        return alerts_df
    
    def create_visualizations(self, sentiment_data: Dict[str, pd.DataFrame],
                            predictions: Dict[str, pd.DataFrame],
                            portfolio_results: Dict,
                            alerts_df: pd.DataFrame,
                            output_dir: str = '/home/claude') -> Dict[str, str]:
        """
        Generate all visualizations
        
        Args:
            sentiment_data: Historical sentiment data
            predictions: Forecast predictions
            portfolio_results: Portfolio optimization results
            alerts_df: Generated alerts
            output_dir: Directory to save visualizations
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        saved_files = {}
        
        print("\nGenerating visualizations...")
        print("-" * 60)
        
        # 1. Time series plots for each company
        for company in list(sentiment_data.keys())[:5]:  # Limit to 5 companies
            pred = predictions.get(company)
            fig = self.visualizer.plot_sentiment_timeseries(
                sentiment_data[company], 
                company, 
                predictions=pred
            )
            filename = f'{output_dir}/ts_{company}.html'
            fig.write_html(filename)
            saved_files[f'timeseries_{company}'] = filename
            print(f"  ✓ Created time series plot: {company}")
        
        # 2. Sentiment heatmap
        fig = self.visualizer.plot_sentiment_heatmap(sentiment_data)
        filename = f'{output_dir}/sentiment_heatmap.html'
        fig.write_html(filename)
        saved_files['heatmap'] = filename
        print(f"  ✓ Created sentiment heatmap")
        
        # 3. Efficient frontier
        fig = self.visualizer.plot_efficient_frontier(
            portfolio_results['efficient_frontier'],
            portfolio_results['optimal_sharpe']
        )
        filename = f'{output_dir}/efficient_frontier.html'
        fig.write_html(filename)
        saved_files['efficient_frontier'] = filename
        print(f"  ✓ Created efficient frontier plot")
        
        # 4. Portfolio allocation
        fig = self.visualizer.plot_portfolio_allocation(
            portfolio_results['optimal_sharpe']['weights'],
            portfolio_results['companies']
        )
        filename = f'{output_dir}/portfolio_allocation.html'
        fig.write_html(filename)
        saved_files['portfolio_allocation'] = filename
        print(f"  ✓ Created portfolio allocation chart")
        
        # 5. Comprehensive dashboard
        fig = self.visualizer.create_dashboard(
            sentiment_data,
            predictions,
            alerts_df,
            portfolio_results
        )
        filename = f'{output_dir}/dashboard.html'
        fig.write_html(filename)
        saved_files['dashboard'] = filename
        print(f"  ✓ Created comprehensive dashboard")
        
        self.results['visualizations'] = saved_files
        return saved_files
    
    def save_results(self, output_dir: str = '/home/claude'):
        """
        Save all results to JSON and CSV files
        
        Args:
            output_dir: Directory to save results
        """
        print("\nSaving results...")
        print("-" * 60)
        
        # Save predictions
        if 'predictions' in self.results:
            for company, pred_df in self.results['predictions'].items():
                if pred_df is not None:
                    filename = f'{output_dir}/predictions_{company}.csv'
                    pred_df.to_csv(filename, index=False)
                    print(f"  ✓ Saved predictions: {company}")
        
        # Save portfolio results
        if 'portfolio' in self.results:
            portfolio = self.results['portfolio']
            
            # Save optimal weights
            weights_df = pd.DataFrame({
                'company': portfolio['companies'],
                'weight': portfolio['optimal_sharpe']['weights'],
                'sentiment_score': [portfolio['sentiment_scores'][c] for c in portfolio['companies']]
            })
            filename = f'{output_dir}/optimal_portfolio.csv'
            weights_df.to_csv(filename, index=False)
            print(f"  ✓ Saved optimal portfolio weights")
            
            # Save efficient frontier
            filename = f'{output_dir}/efficient_frontier.csv'
            portfolio['efficient_frontier'].to_csv(filename, index=False)
            print(f"  ✓ Saved efficient frontier data")
        
        # Save alerts
        if 'alerts' in self.results and len(self.results['alerts']) > 0:
            filename = f'{output_dir}/alerts.csv'
            self.results['alerts'].to_csv(filename, index=False)
            print(f"  ✓ Saved alerts")
        
        # Save summary
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'lookback_days': self.lookback_days,
            'forecast_days': self.forecast_days,
            'companies_analyzed': list(self.results.get('predictions', {}).keys()),
            'total_alerts': len(self.results.get('alerts', [])),
            'portfolio_sharpe_ratio': self.results.get('portfolio', {}).get('optimal_sharpe', {}).get('sharpe_ratio'),
            'portfolio_return': self.results.get('portfolio', {}).get('optimal_sharpe', {}).get('expected_return'),
            'portfolio_volatility': self.results.get('portfolio', {}).get('optimal_sharpe', {}).get('volatility')
        }
        
        filename = f'{output_dir}/analysis_summary.json'
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Saved analysis summary")
    
    def run_full_analysis(self, 
                         ensemble_output_file: Optional[str] = None,
                         companies: Optional[List[str]] = None,
                         train_models: bool = True) -> Dict:
        """
        Run complete end-to-end analysis pipeline
        
        Args:
            ensemble_output_file: Path to CSV with ensemble sentiment scores
            companies: List of company symbols to analyze
            train_models: Whether to train predictive models
        
        Returns:
            Dictionary with all results
        """
        print("="*80)
        print("NLP MARKET SENTIMENT ANALYSIS - FULL PIPELINE")
        print("="*80)
        
        # Load or generate data
        if ensemble_output_file and os.path.exists(ensemble_output_file):
            print(f"\nLoading ensemble output from: {ensemble_output_file}")
            full_df = self.load_ensemble_output(ensemble_output_file)
            companies = companies or full_df['company'].unique().tolist()
        else:
            print("\nGenerating sample data for demonstration...")
            companies = companies or ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
            full_df = self._generate_sample_data(companies)
        
        # Prepare data by company
        sentiment_data = {}
        for company in companies:
            sentiment_data[company] = self.prepare_company_data(full_df, company)
        
        print(f"\nAnalyzing {len(companies)} companies over {self.lookback_days} days")
        
        # Step 1: Time Series Analysis
        print("\n" + "="*80)
        print("STEP 1: TIME SERIES ANALYSIS & FORECASTING")
        print("="*80)
        predictions = self.run_time_series_analysis(sentiment_data, train_models=train_models)
        
        # Step 2: Portfolio Optimization
        print("\n" + "="*80)
        print("STEP 2: PORTFOLIO OPTIMIZATION")
        print("="*80)
        portfolio_results = self.run_portfolio_optimization(sentiment_data)
        
        # Step 3: Alert Generation
        print("\n" + "="*80)
        print("STEP 3: ALERT GENERATION")
        print("="*80)
        alerts_df = self.generate_alerts(sentiment_data)
        
        # Step 4: Visualizations
        print("\n" + "="*80)
        print("STEP 4: VISUALIZATION GENERATION")
        print("="*80)
        viz_files = self.create_visualizations(
            sentiment_data, predictions, portfolio_results, alerts_df
        )
        
        # Step 5: Save Results
        print("\n" + "="*80)
        print("STEP 5: SAVING RESULTS")
        print("="*80)
        self.save_results()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nKey Outputs:")
        print(f"  • Predictions: {len([p for p in predictions.values() if p is not None])} companies")
        print(f"  • Alerts: {len(alerts_df)} detected")
        print(f"  • Visualizations: {len(viz_files)} files created")
        print(f"  • Portfolio Sharpe Ratio: {portfolio_results['optimal_sharpe']['sharpe_ratio']:.4f}")
        
        return self.results
    
    def _generate_sample_data(self, companies: List[str]) -> pd.DataFrame:
        """Generate sample sentiment data for demonstration"""
        np.random.seed(42)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        all_data = []
        
        for company in companies:
            # Generate random walk sentiment
            sentiment_scores = np.random.randn(len(dates)).cumsum() * 0.05
            sentiment_scores = np.clip(sentiment_scores, -1, 1)
            
            for i, date in enumerate(dates):
                all_data.append({
                    'date': date,
                    'company': company,
                    'sentiment_score': sentiment_scores[i],
                    'confidence': np.random.uniform(0.6, 0.95),
                    'volume': np.random.randint(100, 1000),
                    'textblob_score': sentiment_scores[i] + np.random.randn() * 0.1,
                    'finbert_score': sentiment_scores[i] + np.random.randn() * 0.1,
                    'keyword_score': sentiment_scores[i] + np.random.randn() * 0.1,
                    'regime_score': np.random.uniform(-0.5, 0.5),
                    'summary_score': sentiment_scores[i] + np.random.randn() * 0.1,
                    'temporal_score': sentiment_scores[i] + np.random.randn() * 0.1
                })
        
        return pd.DataFrame(all_data)


def main():
    """Example usage of the integration pipeline"""
    
    # Create pipeline
    pipeline = SentimentPipeline(config={
        'lookback_days': 60,
        'forecast_days': 7,
        'risk_free_rate': 0.02
    })
    
    # Run full analysis
    # Option 1: With your actual ensemble output
    # results = pipeline.run_full_analysis(
    #     ensemble_output_file='path/to/your/ensemble_output.csv',
    #     train_models=True
    # )
    
    # Option 2: With sample data for demonstration
    results = pipeline.run_full_analysis(
        companies=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
        train_models=True
    )
    
    print("\n✅ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
