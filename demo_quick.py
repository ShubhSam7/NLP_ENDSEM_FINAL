#!/usr/bin/env python3
"""
Quick Demo Script - NLP Market Sentiment Analysis Future Work

This script provides a quick demonstration of all the new features
without requiring the full pipeline execution.

Run with: python demo_quick.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("NLP MARKET SENTIMENT ANALYSIS - QUICK DEMO")
print("="*80)

# Check dependencies
print("\n1. Checking dependencies...")
print("-" * 80)

dependencies = {
    'numpy': True,
    'pandas': True,
    'scipy': False,
    'torch': False,
    'plotly': False,
    'cvxpy': False
}

try:
    import scipy
    dependencies['scipy'] = True
except ImportError:
    pass

try:
    import torch
    dependencies['torch'] = True
except ImportError:
    pass

try:
    import plotly
    dependencies['plotly'] = True
except ImportError:
    pass

try:
    import cvxpy
    dependencies['cvxpy'] = True
except ImportError:
    pass

for lib, available in dependencies.items():
    status = "✓" if available else "✗"
    print(f"  {status} {lib}")

print("\nCore features available:", all([dependencies[lib] for lib in ['numpy', 'pandas']]))
print("Deep learning available:", dependencies['torch'])
print("Optimization available:", dependencies['scipy'])
print("Visualization available:", dependencies['plotly'])

# Generate sample data
print("\n2. Generating sample sentiment data...")
print("-" * 80)

np.random.seed(42)
companies = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
days = 60
end_date = datetime.now()
start_date = end_date - timedelta(days=days)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

sentiment_data = {}
for company in companies:
    sentiment = np.random.randn(len(dates)).cumsum() * 0.05
    sentiment = np.clip(sentiment, -1, 1)
    
    sentiment_data[company] = pd.DataFrame({
        'date': dates,
        'sentiment_score': sentiment,
        'confidence': np.random.uniform(0.6, 0.95, len(dates)),
        'volume': np.random.randint(100, 1000, len(dates))
    })

print(f"Generated {days} days of sentiment data for {len(companies)} companies")
print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

# Display current sentiment
print("\nCurrent Sentiment Scores:")
for company, df in sentiment_data.items():
    score = df['sentiment_score'].iloc[-1]
    sentiment_label = "Bullish" if score > 0.2 else "Bearish" if score < -0.2 else "Neutral"
    print(f"  {company}: {score:+.3f} ({sentiment_label})")

# Time series analysis (if PyTorch available)
if dependencies['torch']:
    print("\n3. Time Series Analysis & Forecasting")
    print("-" * 80)
    
    from future_work_implementation import TimeSeriesAnalyzer
    
    analyzer = TimeSeriesAnalyzer(sequence_length=10)
    company = 'AAPL'
    
    print(f"Training LSTM model for {company}...")
    history = analyzer.train_lstm_model(
        sentiment_data[company],
        epochs=20,
        batch_size=16,
        learning_rate=0.001
    )
    
    print(f"  Final validation loss: {history['val_loss'][-1]:.6f}")
    
    predictions = analyzer.predict_future_sentiment(sentiment_data[company], days_ahead=7)
    print(f"\n7-Day Forecast for {company}:")
    for _, row in predictions.head().iterrows():
        print(f"  {row['date'].strftime('%Y-%m-%d')}: {row['predicted_sentiment']:+.3f}")
else:
    print("\n3. Time Series Analysis - SKIPPED (PyTorch not available)")
    print("-" * 80)
    print("  Install PyTorch to enable deep learning forecasting")

# Portfolio optimization (if scipy available)
if dependencies['scipy']:
    print("\n4. Portfolio Optimization")
    print("-" * 80)
    
    from future_work_implementation import PortfolioOptimizer
    
    optimizer = PortfolioOptimizer(risk_free_rate=0.02)
    
    # Get latest sentiment scores
    sentiment_scores = {
        company: df['sentiment_score'].iloc[-1]
        for company, df in sentiment_data.items()
    }
    
    # Generate sample returns
    returns_data = pd.DataFrame({
        company: np.random.randn(len(dates)) * 0.02 + 0.001
        for company in companies
    })
    
    expected_returns = optimizer.calculate_sentiment_weighted_returns(
        sentiment_scores, returns_data
    )
    
    cov_matrix = returns_data.cov().values
    
    optimal = optimizer.optimize_portfolio_sharpe(expected_returns, cov_matrix)
    
    print("Optimal Portfolio (Maximum Sharpe):")
    for i, company in enumerate(companies):
        weight = optimal['weights'][i]
        if weight > 0.01:
            print(f"  {company:6s}: {weight:6.2%} (Sentiment: {sentiment_scores[company]:+.3f})")
    
    print(f"\nExpected Return: {optimal['expected_return']:.4f}")
    print(f"Volatility:      {optimal['volatility']:.4f}")
    print(f"Sharpe Ratio:    {optimal['sharpe_ratio']:.4f}")
else:
    print("\n4. Portfolio Optimization - SKIPPED (scipy not available)")
    print("-" * 80)
    print("  Install scipy to enable portfolio optimization")

# Alert system
print("\n5. Real-time Alert System")
print("-" * 80)

from future_work_implementation import SentimentAlertSystem

alert_system = SentimentAlertSystem(lookback_period=30, z_threshold=2.0)
alerts_df = alert_system.get_all_alerts(sentiment_data)

if len(alerts_df) > 0:
    print(f"Generated {len(alerts_df)} alerts:")
    for _, alert in alerts_df.head(5).iterrows():
        severity_icon = "🔴" if alert.get('severity') == 'high' else "🟡"
        alert_type = alert.get('alert_type', alert.get('reversal_type', 'unknown'))
        company = alert.get('company', 'N/A')
        print(f"  {severity_icon} {company}: {alert_type}")
else:
    print("No alerts detected (market sentiment is stable)")

# Visualization (if plotly available)
if dependencies['plotly']:
    print("\n6. Generating Visualizations")
    print("-" * 80)
    
    from future_work_implementation import EnhancedVisualizer
    
    visualizer = EnhancedVisualizer()
    
    # Time series plot
    company = 'AAPL'
    pred = predictions if dependencies['torch'] else None
    fig = visualizer.plot_sentiment_timeseries(
        sentiment_data[company],
        company,
        predictions=pred
    )
    
    filename = '/home/claude/demo_timeseries.html'
    fig.write_html(filename)
    print(f"  ✓ Created: {filename}")
    
    # Heatmap
    fig = visualizer.plot_sentiment_heatmap(sentiment_data)
    filename = '/home/claude/demo_heatmap.html'
    fig.write_html(filename)
    print(f"  ✓ Created: {filename}")
    
    # Portfolio visualizations
    if dependencies['scipy']:
        fig = visualizer.plot_portfolio_allocation(optimal['weights'], companies)
        filename = '/home/claude/demo_allocation.html'
        fig.write_html(filename)
        print(f"  ✓ Created: {filename}")
else:
    print("\n6. Visualizations - SKIPPED (plotly not available)")
    print("-" * 80)
    print("  Install plotly to enable interactive visualizations")

# Summary
print("\n" + "="*80)
print("DEMO COMPLETE")
print("="*80)

print("\n✅ Successfully demonstrated:")
features_shown = ["✓ Data generation and preprocessing"]

if dependencies['torch']:
    features_shown.append("✓ LSTM time series forecasting")
if dependencies['scipy']:
    features_shown.append("✓ Portfolio optimization")
features_shown.append("✓ Alert system")
if dependencies['plotly']:
    features_shown.append("✓ Interactive visualizations")

for feature in features_shown:
    print(f"  {feature}")

print("\n📚 Next Steps:")
print("  1. Install missing dependencies: pip install -r requirements_future_work.txt")
print("  2. Integrate with your ensemble output: use integration.py")
print("  3. Run full pipeline: python integration.py")
print("  4. Explore Jupyter notebook: demo_notebook.ipynb")
print("\n📖 Documentation: README_FUTURE_WORK.md")

print("\n" + "="*80)
