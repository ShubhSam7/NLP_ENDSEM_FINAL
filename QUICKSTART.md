# Quick Start Guide - Get Running in 5 Minutes

## Step 1: Copy Files to Your Repository (1 minute)

```bash
cd /path/to/NLP_market_sentiment_analysis

# Create a new directory for future work
mkdir future_work
cd future_work

# Copy these files from the deliverables
cp /path/to/deliverables/* .
```

## Step 2: Install Dependencies (2 minutes)

```bash
# Install core dependencies (required)
pip install numpy pandas scipy scikit-learn

# Install deep learning (optional, for LSTM forecasting)
pip install torch torchvision

# Install visualization (optional, recommended)
pip install plotly matplotlib seaborn

# Install optimization (optional, for advanced portfolio features)
pip install cvxpy

# OR install everything at once
pip install -r requirements_future_work.txt
```

## Step 3: Test the Installation (1 minute)

```bash
# Run the quick demo
python demo_quick.py

# You should see:
# ✓ Data generation
# ✓ Portfolio optimization  
# ✓ Alert system
# ✓ (Optional) Forecasting if PyTorch installed
# ✓ (Optional) Visualizations if Plotly installed
```

## Step 4: Connect to Your Ensemble (1 minute)

### Option A: Use the Integration Pipeline

```python
from integration import SentimentPipeline

pipeline = SentimentPipeline()

# Point to your ensemble output CSV
results = pipeline.run_full_analysis(
    ensemble_output_file='../data/ensemble_sentiment_output.csv'
)
```

### Option B: Use Individual Components

```python
from future_work_implementation import (
    TimeSeriesAnalyzer,
    PortfolioOptimizer,
    SentimentAlertSystem
)
import pandas as pd

# Load your sentiment data
df = pd.read_csv('your_sentiment_data.csv')
df['date'] = pd.to_datetime(df['date'])

# 1. Forecast sentiment
analyzer = TimeSeriesAnalyzer()
predictions = analyzer.predict_future_sentiment(df, days_ahead=7)

# 2. Optimize portfolio
optimizer = PortfolioOptimizer()
# ... (see full examples in README_FUTURE_WORK.md)

# 3. Generate alerts
alert_system = SentimentAlertSystem()
alerts = alert_system.get_all_alerts(sentiment_data_dict)
```

## Expected CSV Format

Your ensemble system should output CSV files like this:

```csv
date,company,sentiment_score,confidence
2024-10-01,AAPL,0.65,0.85
2024-10-01,GOOGL,0.42,0.78
2024-10-02,AAPL,0.58,0.82
2024-10-02,GOOGL,0.45,0.80
```

**Required columns:**
- `date`: Date in YYYY-MM-DD format
- `company`: Stock ticker
- `sentiment_score`: Score from -1 to +1
- `confidence`: Confidence from 0 to 1

## Common Use Cases

### Use Case 1: Daily Sentiment Forecast

```python
from integration import SentimentPipeline

pipeline = SentimentPipeline(config={
    'forecast_days': 7,
    'lookback_days': 30
})

# Run daily
results = pipeline.run_full_analysis(
    ensemble_output_file='daily_sentiment.csv'
)

# Get tomorrow's predicted sentiment
tomorrow = results['predictions']['AAPL'].iloc[0]
print(f"Tomorrow's AAPL sentiment: {tomorrow['predicted_sentiment']:.3f}")
```

### Use Case 2: Portfolio Rebalancing

```python
from future_work_implementation import PortfolioOptimizer
import pandas as pd

# Get current sentiment for your watchlist
sentiment_scores = {
    'AAPL': 0.65,
    'GOOGL': 0.42,
    'MSFT': 0.58,
    'TSLA': -0.23
}

# Load historical returns
returns = pd.read_csv('returns.csv')

# Optimize
optimizer = PortfolioOptimizer(risk_free_rate=0.02)
expected_returns = optimizer.calculate_sentiment_weighted_returns(
    sentiment_scores, returns
)
optimal = optimizer.optimize_portfolio_sharpe(
    expected_returns, returns.cov().values
)

# Print allocation
for company, weight in zip(sentiment_scores.keys(), optimal['weights']):
    print(f"{company}: {weight:.1%}")
```

### Use Case 3: Real-time Monitoring

```python
from future_work_implementation import SentimentAlertSystem

alert_system = SentimentAlertSystem(z_threshold=2.0)

# Run every hour/day
alerts = alert_system.get_all_alerts(sentiment_data_dict)

# Send notifications for high-severity alerts
high_alerts = alerts[alerts['severity'] == 'high']
for _, alert in high_alerts.iterrows():
    send_notification(
        f"Alert: {alert['company']} - {alert['alert_type']}"
    )
```

## Output Files

After running the pipeline, you'll find:

```
outputs/
├── predictions_AAPL.csv          # 7-day forecast
├── predictions_GOOGL.csv
├── optimal_portfolio.csv         # Weights and allocations
├── efficient_frontier.csv        # Risk-return tradeoff data
├── alerts.csv                    # Detected anomalies
├── sentiment_heatmap.html        # Interactive visualization
├── ts_AAPL.html                  # Time series plot
└── dashboard.html                # Comprehensive view
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
**Fix**: PyTorch is optional. The portfolio optimization and alerts work without it.
```bash
pip install torch  # If you want forecasting features
```

### "ImportError: Scipy is required"
**Fix**: Install scipy for portfolio optimization.
```bash
pip install scipy
```

### "No alerts detected"
**Normal**: This means sentiment is stable. Adjust threshold to be more sensitive:
```python
alert_system = SentimentAlertSystem(z_threshold=1.5)  # Lower = more sensitive
```

### "Optimization not converging"
**Fix**: Relax constraints or simplify the problem:
```python
constraints = {'min_weight': 0.0, 'max_weight': 1.0}
```

## Next Steps

1. ✅ Run demo_quick.py successfully
2. ✅ Test with sample data
3. ✅ Integrate with your ensemble output
4. ✅ Generate visualizations
5. ✅ Set up automated alerts
6. 📚 Read README_FUTURE_WORK.md for advanced features
7. 🔬 Experiment with demo_notebook.ipynb
8. 📊 Backtest strategies on historical data

## Getting Help

- **Documentation**: README_FUTURE_WORK.md (comprehensive guide)
- **Examples**: demo_notebook.ipynb (interactive tutorial)
- **Quick Test**: demo_quick.py (verifies installation)
- **GitHub Issues**: For bugs and feature requests
- **Code Comments**: All functions have detailed docstrings

## Performance Tips

1. **Use GPU for LSTM Training** (5x faster)
   ```python
   import torch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **Cache Model Weights** (avoid retraining)
   ```python
   torch.save(analyzer.model.state_dict(), 'model.pt')
   analyzer.model.load_state_dict(torch.load('model.pt'))
   ```

3. **Batch Process Multiple Companies**
   ```python
   results = pipeline.run_full_analysis(
       companies=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
   )
   ```

4. **Limit History for Speed**
   ```python
   df_recent = df.tail(60)  # Last 60 days only
   ```

## That's It!

You now have a complete advanced analytics system running on top of your sentiment analysis. The entire setup should take less than 5 minutes.

For detailed information on each feature, see README_FUTURE_WORK.md.

---

**Questions?** Check PROJECT_SUMMARY.md for technical details and architecture overview.
