"""
Future Work: Advanced Market Sentiment Analysis Extensions
============================================================
This module extends the FinLlama ensemble sentiment analyzer with:
1. Time Series Analysis & Forecasting
2. Portfolio Optimization Engine
3. Real-time Alert System
4. Enhanced Visualization Dashboard

Author: NLP Market Sentiment Analysis Team
Date: November 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Deep Learning & ML
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Deep learning features will be limited.")
    # Create dummy classes to prevent NameError
    class Dataset:
        pass
    class nn:
        class Module:
            pass

try:
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available.")

# Optimization
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: Scipy not available.")

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: CVXPY not available.")

OPTIMIZATION_AVAILABLE = SCIPY_AVAILABLE or CVXPY_AVAILABLE

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False
    print("Warning: Visualization libraries not available.")


class SentimentTimeSeriesDataset(Dataset):
    """PyTorch Dataset for sentiment time series data"""
    
    def __init__(self, data: np.ndarray, sequence_length: int = 10):
        """
        Args:
            data: Array of sentiment scores shape (n_samples, n_features)
            sequence_length: Number of past days to use for prediction
        """
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        X = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, 0]  # Predict next day sentiment
        return X, y


class LSTMSentimentPredictor(nn.Module if TORCH_AVAILABLE else object):
    """LSTM model for sentiment forecasting"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM models")
        
        super(LSTMSentimentPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_output = lstm_out[:, -1, :]
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class TimeSeriesAnalyzer:
    """Advanced time series analysis for sentiment data"""
    
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        if SKLEARN_AVAILABLE:
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
        else:
            self.scaler = None
        self.model = None
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = None
        
    def prepare_data(self, sentiment_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sentiment data for training
        
        Args:
            sentiment_df: DataFrame with columns ['date', 'sentiment_score', 'confidence', 'volume']
        
        Returns:
            X_train, y_train arrays
        """
        # Ensure data is sorted by date
        sentiment_df = sentiment_df.sort_values('date').reset_index(drop=True)
        
        # Create features
        features = ['sentiment_score']
        if 'confidence' in sentiment_df.columns:
            features.append('confidence')
        if 'volume' in sentiment_df.columns:
            features.append('volume')
            
        # Add technical indicators
        sentiment_df['ma_7'] = sentiment_df['sentiment_score'].rolling(window=7, min_periods=1).mean()
        sentiment_df['ma_30'] = sentiment_df['sentiment_score'].rolling(window=30, min_periods=1).mean()
        sentiment_df['std_7'] = sentiment_df['sentiment_score'].rolling(window=7, min_periods=1).std()
        sentiment_df['momentum'] = sentiment_df['sentiment_score'].diff()
        
        features.extend(['ma_7', 'ma_30', 'std_7', 'momentum'])
        
        # Fill NaN values
        sentiment_df[features] = sentiment_df[features].fillna(method='bfill').fillna(0)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(sentiment_df[features].values)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length, 0])  # Predict sentiment_score
            
        return np.array(X), np.array(y)
    
    def train_lstm_model(self, sentiment_df: pd.DataFrame, epochs: int = 50, batch_size: int = 32, 
                        learning_rate: float = 0.001) -> Dict:
        """
        Train LSTM model for sentiment forecasting
        
        Returns:
            Dictionary with training history
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM training")
        
        X, y = self.prepare_data(sentiment_df)
        
        # Split data (80-20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets and dataloaders
        train_dataset = SentimentTimeSeriesDataset(
            np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
        )
        val_dataset = SentimentTimeSeriesDataset(
            np.concatenate([X_val, y_val.reshape(-1, 1)], axis=1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X.shape[2]
        self.model = LSTMSentimentPredictor(input_size=input_size).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device).unsqueeze(1)
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), '/home/claude/best_lstm_model.pt')
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        return history
    
    def predict_future_sentiment(self, sentiment_df: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """
        Predict future sentiment scores
        
        Args:
            sentiment_df: Historical sentiment data
            days_ahead: Number of days to forecast
        
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_lstm_model first.")
        
        self.model.eval()
        predictions = []
        
        # Prepare last sequence
        X, _ = self.prepare_data(sentiment_df)
        last_sequence = torch.FloatTensor(X[-1]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(days_ahead):
                pred = self.model(last_sequence)
                predictions.append(pred.cpu().numpy()[0, 0])
                
                # Update sequence for next prediction
                new_point = np.zeros((1, 1, last_sequence.shape[2]))
                new_point[0, 0, 0] = pred.cpu().numpy()[0, 0]
                last_sequence = torch.cat([last_sequence[:, 1:, :], 
                                          torch.FloatTensor(new_point).to(self.device)], dim=1)
        
        # Inverse transform predictions
        pred_array = np.array(predictions).reshape(-1, 1)
        # Create dummy array with right shape for inverse transform
        dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy[:, 0] = predictions
        unscaled = self.scaler.inverse_transform(dummy)[:, 0]
        
        # Create prediction dataframe
        last_date = pd.to_datetime(sentiment_df['date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        pred_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sentiment': unscaled,
            'prediction_type': 'forecast'
        })
        
        return pred_df
    
    def calculate_prediction_intervals(self, sentiment_df: pd.DataFrame, 
                                      days_ahead: int = 7, 
                                      confidence: float = 0.95) -> pd.DataFrame:
        """Calculate prediction intervals using bootstrap method"""
        predictions = []
        
        # Generate multiple predictions with noise
        for _ in range(100):
            pred_df = self.predict_future_sentiment(sentiment_df, days_ahead)
            predictions.append(pred_df['predicted_sentiment'].values)
        
        predictions = np.array(predictions)
        
        # Calculate confidence intervals
        lower_bound = np.percentile(predictions, (1 - confidence) * 50, axis=0)
        upper_bound = np.percentile(predictions, (1 + confidence) * 50, axis=0)
        mean_pred = np.mean(predictions, axis=0)
        
        last_date = pd.to_datetime(sentiment_df['date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
        
        return pd.DataFrame({
            'date': future_dates,
            'predicted_sentiment': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        })


class PortfolioOptimizer:
    """Portfolio optimization using sentiment signals"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        
    def calculate_sentiment_weighted_returns(self, 
                                            sentiment_scores: Dict[str, float],
                                            historical_returns: pd.DataFrame) -> np.ndarray:
        """
        Calculate expected returns adjusted by sentiment
        
        Args:
            sentiment_scores: Dict mapping stock symbols to sentiment scores [-1, 1]
            historical_returns: DataFrame with stock returns
        
        Returns:
            Array of expected returns
        """
        expected_returns = historical_returns.mean().values
        
        # Adjust returns based on sentiment
        for i, symbol in enumerate(historical_returns.columns):
            if symbol in sentiment_scores:
                sentiment = sentiment_scores[symbol]
                # Positive sentiment increases expected return, negative decreases
                adjustment_factor = 1 + (sentiment * 0.2)  # 20% max adjustment
                expected_returns[i] *= adjustment_factor
        
        return expected_returns
    
    def optimize_portfolio_sharpe(self, 
                                  expected_returns: np.ndarray,
                                  cov_matrix: np.ndarray,
                                  constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio for maximum Sharpe ratio
        
        Args:
            expected_returns: Array of expected returns
            cov_matrix: Covariance matrix of returns
            constraints: Dict with 'min_weight' and 'max_weight'
        
        Returns:
            Dictionary with optimal weights and performance metrics
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("Scipy is required for portfolio optimization. Install with: pip install scipy")
        
        n_assets = len(expected_returns)
        
        # Set constraints
        if constraints is None:
            constraints = {'min_weight': 0.0, 'max_weight': 0.3}
        
        def neg_sharpe(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe
        
        # Constraints: weights sum to 1, and within bounds
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets))
        constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Initial guess: equal weights
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_std,
            'sharpe_ratio': sharpe_ratio
        }
    
    def optimize_portfolio_min_variance(self, 
                                       cov_matrix: np.ndarray,
                                       constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio for minimum variance
        
        Returns:
            Dictionary with optimal weights and risk metrics
        """
        if not SCIPY_AVAILABLE:
            raise ImportError("Scipy is required for portfolio optimization. Install with: pip install scipy")
        
        n_assets = cov_matrix.shape[0]
        
        if constraints is None:
            constraints = {'min_weight': 0.0, 'max_weight': 0.3}
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        bounds = tuple((constraints['min_weight'], constraints['max_weight']) for _ in range(n_assets))
        constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        portfolio_variance_val = portfolio_variance(optimal_weights)
        
        return {
            'weights': optimal_weights,
            'variance': portfolio_variance_val,
            'volatility': np.sqrt(portfolio_variance_val)
        }
    
    def generate_efficient_frontier(self, 
                                   expected_returns: np.ndarray,
                                   cov_matrix: np.ndarray,
                                   n_points: int = 100) -> pd.DataFrame:
        """
        Generate efficient frontier data points
        
        Returns:
            DataFrame with returns, volatility, and weights for each point
        """
        results = []
        target_returns = np.linspace(expected_returns.min(), expected_returns.max(), n_points)
        
        n_assets = len(expected_returns)
        
        for target in target_returns:
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                {'type': 'eq', 'fun': lambda w: np.sum(w * expected_returns) - target}
            ]
            
            bounds = tuple((0, 1) for _ in range(n_assets))
            initial_weights = np.array([1/n_assets] * n_assets)
            
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                volatility = np.sqrt(result.fun)
                results.append({
                    'return': target,
                    'volatility': volatility,
                    'sharpe': (target - self.risk_free_rate) / volatility,
                    'weights': result.x
                })
        
        return pd.DataFrame(results)


class SentimentAlertSystem:
    """Real-time alert system for sentiment anomalies"""
    
    def __init__(self, lookback_period: int = 30, z_threshold: float = 2.0):
        self.lookback_period = lookback_period
        self.z_threshold = z_threshold
        self.alert_history = []
        
    def detect_sentiment_spike(self, sentiment_df: pd.DataFrame, company: str) -> Optional[Dict]:
        """
        Detect unusual sentiment spikes
        
        Args:
            sentiment_df: Historical sentiment data
            company: Company symbol
        
        Returns:
            Alert dictionary if spike detected, None otherwise
        """
        if len(sentiment_df) < self.lookback_period:
            return None
        
        recent_data = sentiment_df.tail(self.lookback_period)
        current_sentiment = sentiment_df['sentiment_score'].iloc[-1]
        
        # Calculate rolling statistics
        mean = recent_data['sentiment_score'].mean()
        std = recent_data['sentiment_score'].std()
        
        if std == 0:
            return None
        
        z_score = (current_sentiment - mean) / std
        
        if abs(z_score) > self.z_threshold:
            alert = {
                'company': company,
                'date': sentiment_df['date'].iloc[-1],
                'current_sentiment': current_sentiment,
                'mean_sentiment': mean,
                'z_score': z_score,
                'alert_type': 'spike_positive' if z_score > 0 else 'spike_negative',
                'severity': 'high' if abs(z_score) > 3 else 'medium'
            }
            self.alert_history.append(alert)
            return alert
        
        return None
    
    def detect_trend_reversal(self, sentiment_df: pd.DataFrame, company: str, 
                             window: int = 7) -> Optional[Dict]:
        """
        Detect sentiment trend reversals
        
        Args:
            sentiment_df: Historical sentiment data
            company: Company symbol
            window: Window size for trend calculation
        
        Returns:
            Alert dictionary if reversal detected
        """
        if len(sentiment_df) < window * 2:
            return None
        
        recent = sentiment_df['sentiment_score'].tail(window).mean()
        previous = sentiment_df['sentiment_score'].tail(window * 2).head(window).mean()
        
        # Check for significant reversal
        change = recent - previous
        pct_change = abs(change / previous) if previous != 0 else 0
        
        if pct_change > 0.5:  # 50% change
            alert = {
                'company': company,
                'date': sentiment_df['date'].iloc[-1],
                'reversal_type': 'bullish_reversal' if change > 0 else 'bearish_reversal',
                'previous_sentiment': previous,
                'current_sentiment': recent,
                'pct_change': pct_change * 100,
                'severity': 'high' if pct_change > 1.0 else 'medium'
            }
            self.alert_history.append(alert)
            return alert
        
        return None
    
    def detect_volatility_surge(self, sentiment_df: pd.DataFrame, company: str) -> Optional[Dict]:
        """
        Detect unusual increases in sentiment volatility
        
        Args:
            sentiment_df: Historical sentiment data
            company: Company symbol
        
        Returns:
            Alert dictionary if volatility surge detected
        """
        if len(sentiment_df) < self.lookback_period * 2:
            return None
        
        recent_vol = sentiment_df['sentiment_score'].tail(self.lookback_period).std()
        historical_vol = sentiment_df['sentiment_score'].tail(self.lookback_period * 2).head(self.lookback_period).std()
        
        if historical_vol == 0:
            return None
        
        vol_ratio = recent_vol / historical_vol
        
        if vol_ratio > 2.0:  # Volatility doubled
            alert = {
                'company': company,
                'date': sentiment_df['date'].iloc[-1],
                'alert_type': 'volatility_surge',
                'recent_volatility': recent_vol,
                'historical_volatility': historical_vol,
                'vol_ratio': vol_ratio,
                'severity': 'high' if vol_ratio > 3.0 else 'medium'
            }
            self.alert_history.append(alert)
            return alert
        
        return None
    
    def get_all_alerts(self, sentiment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate all alerts for multiple companies
        
        Args:
            sentiment_data: Dictionary mapping company symbols to sentiment DataFrames
        
        Returns:
            DataFrame with all alerts
        """
        all_alerts = []
        
        for company, df in sentiment_data.items():
            # Check for spikes
            spike_alert = self.detect_sentiment_spike(df, company)
            if spike_alert:
                all_alerts.append(spike_alert)
            
            # Check for reversals
            reversal_alert = self.detect_trend_reversal(df, company)
            if reversal_alert:
                all_alerts.append(reversal_alert)
            
            # Check for volatility surges
            vol_alert = self.detect_volatility_surge(df, company)
            if vol_alert:
                all_alerts.append(vol_alert)
        
        if not all_alerts:
            return pd.DataFrame()
        
        return pd.DataFrame(all_alerts)


class EnhancedVisualizer:
    """Enhanced visualization dashboard for sentiment analysis"""
    
    @staticmethod
    def plot_sentiment_timeseries(sentiment_df: pd.DataFrame, company: str, 
                                  predictions: Optional[pd.DataFrame] = None):
        """Plot sentiment time series with optional predictions"""
        if not VIZ_AVAILABLE:
            print("Visualization libraries not available")
            return
        
        fig = go.Figure()
        
        # Historical sentiment
        fig.add_trace(go.Scatter(
            x=sentiment_df['date'],
            y=sentiment_df['sentiment_score'],
            mode='lines+markers',
            name='Historical Sentiment',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Add confidence bands if available
        if 'confidence' in sentiment_df.columns:
            fig.add_trace(go.Scatter(
                x=sentiment_df['date'],
                y=sentiment_df['sentiment_score'] + sentiment_df['confidence'],
                mode='lines',
                name='Upper Confidence',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=sentiment_df['date'],
                y=sentiment_df['sentiment_score'] - sentiment_df['confidence'],
                mode='lines',
                name='Lower Confidence',
                line=dict(width=0),
                fillcolor='rgba(0, 100, 255, 0.2)',
                fill='tonexty',
                showlegend=True
            ))
        
        # Predictions
        if predictions is not None:
            fig.add_trace(go.Scatter(
                x=predictions['date'],
                y=predictions['predicted_sentiment'],
                mode='lines+markers',
                name='Predicted Sentiment',
                line=dict(color='red', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            if 'lower_bound' in predictions.columns:
                fig.add_trace(go.Scatter(
                    x=predictions['date'],
                    y=predictions['upper_bound'],
                    mode='lines',
                    name='Prediction Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=predictions['date'],
                    y=predictions['lower_bound'],
                    mode='lines',
                    name='Prediction Lower Bound',
                    line=dict(width=0),
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    fill='tonexty',
                    showlegend=True
                ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f'Sentiment Analysis for {company}',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            yaxis=dict(range=[-1.2, 1.2]),
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_sentiment_heatmap(sentiment_data: Dict[str, pd.DataFrame], 
                              start_date: Optional[str] = None):
        """Create heatmap of sentiment across multiple companies"""
        if not VIZ_AVAILABLE:
            print("Visualization libraries not available")
            return
        
        # Prepare data matrix
        companies = list(sentiment_data.keys())
        
        # Find common date range
        if start_date:
            start_date = pd.to_datetime(start_date)
        else:
            start_date = max([df['date'].min() for df in sentiment_data.values()])
        
        end_date = min([df['date'].max() for df in sentiment_data.values()])
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create matrix
        matrix = []
        for company in companies:
            df = sentiment_data[company]
            df = df.set_index('date')
            df = df.reindex(date_range, fill_value=0)
            matrix.append(df['sentiment_score'].values)
        
        matrix = np.array(matrix)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[d.strftime('%Y-%m-%d') for d in date_range],
            y=companies,
            colorscale='RdYlGn',
            zmid=0,
            colorbar=dict(title='Sentiment Score')
        ))
        
        fig.update_layout(
            title='Sentiment Heatmap Across Companies',
            xaxis_title='Date',
            yaxis_title='Company',
            height=max(400, len(companies) * 30),
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def plot_efficient_frontier(frontier_df: pd.DataFrame, optimal_portfolio: Dict):
        """Plot efficient frontier with optimal portfolio"""
        if not VIZ_AVAILABLE:
            print("Visualization libraries not available")
            return
        
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_df['volatility'],
            y=frontier_df['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='blue', width=3)
        ))
        
        # Color by Sharpe ratio
        fig.add_trace(go.Scatter(
            x=frontier_df['volatility'],
            y=frontier_df['return'],
            mode='markers',
            name='Sharpe Ratio',
            marker=dict(
                size=10,
                color=frontier_df['sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            ),
            showlegend=False
        ))
        
        # Optimal portfolio
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio['volatility']],
            y=[optimal_portfolio['expected_return']],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(size=15, color='red', symbol='star')
        ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Portfolio Volatility (Risk)',
            yaxis_title='Expected Return',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_portfolio_allocation(weights: np.ndarray, symbols: List[str]):
        """Plot portfolio allocation pie chart"""
        if not VIZ_AVAILABLE:
            print("Visualization libraries not available")
            return
        
        # Filter out zero weights
        non_zero_idx = weights > 0.01
        filtered_weights = weights[non_zero_idx]
        filtered_symbols = [s for i, s in enumerate(symbols) if non_zero_idx[i]]
        
        fig = go.Figure(data=[go.Pie(
            labels=filtered_symbols,
            values=filtered_weights,
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title='Optimal Portfolio Allocation',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_dashboard(sentiment_data: Dict[str, pd.DataFrame],
                        predictions: Dict[str, pd.DataFrame],
                        alerts_df: pd.DataFrame,
                        portfolio_results: Dict):
        """Create comprehensive interactive dashboard"""
        if not VIZ_AVAILABLE:
            print("Visualization libraries not available")
            return
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        companies = list(sentiment_data.keys())[:4]  # Limit to 4 for display
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=companies,
            specs=[[{'type': 'scatter'}] * 2] * 2
        )
        
        for idx, company in enumerate(companies):
            row = (idx // 2) + 1
            col = (idx % 2) + 1
            
            df = sentiment_data[company]
            pred = predictions.get(company)
            
            fig.add_trace(
                go.Scatter(
                    x=df['date'],
                    y=df['sentiment_score'],
                    mode='lines',
                    name=f'{company} Historical',
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            if pred is not None:
                fig.add_trace(
                    go.Scatter(
                        x=pred['date'],
                        y=pred['predicted_sentiment'],
                        mode='lines',
                        name=f'{company} Predicted',
                        line=dict(dash='dash'),
                        showlegend=(idx == 0)
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title_text='Multi-Company Sentiment Dashboard',
            height=800,
            template='plotly_white',
            showlegend=True
        )
        
        return fig


# Example usage and integration functions
def demonstrate_future_work():
    """
    Demonstration of all future work components
    """
    print("="*80)
    print("NLP Market Sentiment Analysis - Future Work Demonstration")
    print("="*80)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', end='2024-10-31', freq='D')
    
    # Sample sentiment data for multiple companies
    companies = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    sentiment_data = {}
    
    for company in companies:
        base_sentiment = np.random.randn(len(dates)).cumsum() * 0.05
        sentiment_scores = np.clip(base_sentiment, -1, 1)
        
        sentiment_data[company] = pd.DataFrame({
            'date': dates,
            'sentiment_score': sentiment_scores,
            'confidence': np.random.uniform(0.6, 0.95, len(dates)),
            'volume': np.random.randint(100, 1000, len(dates))
        })
    
    print("\n1. Time Series Analysis & Forecasting")
    print("-" * 80)
    
    if TORCH_AVAILABLE:
        analyzer = TimeSeriesAnalyzer(sequence_length=10)
        
        # Train model
        print("Training LSTM model...")
        history = analyzer.train_lstm_model(sentiment_data['AAPL'], epochs=30)
        print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        
        # Make predictions
        predictions = analyzer.predict_future_sentiment(sentiment_data['AAPL'], days_ahead=14)
        print(f"\nPredicted sentiment for next 14 days:")
        print(predictions.head())
        
        # Prediction intervals
        intervals = analyzer.calculate_prediction_intervals(sentiment_data['AAPL'], days_ahead=7)
        print(f"\nPrediction intervals:")
        print(intervals)
    else:
        print("PyTorch not available - skipping LSTM demonstration")
    
    print("\n2. Portfolio Optimization")
    print("-" * 80)
    
    if OPTIMIZATION_AVAILABLE:
        optimizer = PortfolioOptimizer(risk_free_rate=0.02)
        
        # Generate sample returns data
        returns_data = pd.DataFrame({
            company: np.random.randn(len(dates)) * 0.02 + 0.001
            for company in companies
        })
        
        # Get latest sentiment scores
        sentiment_scores = {
            company: df['sentiment_score'].iloc[-1]
            for company, df in sentiment_data.items()
        }
        
        # Calculate sentiment-adjusted returns
        expected_returns = optimizer.calculate_sentiment_weighted_returns(
            sentiment_scores, returns_data
        )
        cov_matrix = returns_data.cov().values
        
        # Optimize portfolio
        optimal_portfolio = optimizer.optimize_portfolio_sharpe(expected_returns, cov_matrix)
        
        print("\nOptimal Portfolio (Max Sharpe):")
        for i, company in enumerate(companies):
            print(f"  {company}: {optimal_portfolio['weights'][i]:.2%}")
        print(f"Expected Return: {optimal_portfolio['expected_return']:.4f}")
        print(f"Volatility: {optimal_portfolio['volatility']:.4f}")
        print(f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.4f}")
        
        # Generate efficient frontier
        frontier_df = optimizer.generate_efficient_frontier(expected_returns, cov_matrix, n_points=50)
        print(f"\nGenerated {len(frontier_df)} points on efficient frontier")
    else:
        print("Optimization libraries not available - skipping portfolio optimization")
    
    print("\n3. Alert System")
    print("-" * 80)
    
    alert_system = SentimentAlertSystem(lookback_period=30, z_threshold=2.0)
    alerts_df = alert_system.get_all_alerts(sentiment_data)
    
    if len(alerts_df) > 0:
        print(f"\nGenerated {len(alerts_df)} alerts:")
        print(alerts_df[['company', 'alert_type', 'severity']].head())
    else:
        print("No alerts detected in sample data")
    
    print("\n4. Enhanced Visualizations")
    print("-" * 80)
    
    if VIZ_AVAILABLE:
        visualizer = EnhancedVisualizer()
        
        # Time series plot
        if TORCH_AVAILABLE:
            fig1 = visualizer.plot_sentiment_timeseries(
                sentiment_data['AAPL'], 
                'AAPL', 
                predictions=predictions
            )
            fig1.write_html('/home/claude/sentiment_timeseries.html')
            print("Created: sentiment_timeseries.html")
        
        # Heatmap
        fig2 = visualizer.plot_sentiment_heatmap(sentiment_data)
        fig2.write_html('/home/claude/sentiment_heatmap.html')
        print("Created: sentiment_heatmap.html")
        
        # Portfolio visualizations
        if OPTIMIZATION_AVAILABLE:
            fig3 = visualizer.plot_efficient_frontier(frontier_df, optimal_portfolio)
            fig3.write_html('/home/claude/efficient_frontier.html')
            print("Created: efficient_frontier.html")
            
            fig4 = visualizer.plot_portfolio_allocation(optimal_portfolio['weights'], companies)
            fig4.write_html('/home/claude/portfolio_allocation.html')
            print("Created: portfolio_allocation.html")
        
        print("\nAll visualizations saved to /home/claude/")
    else:
        print("Visualization libraries not available")
    
    print("\n" + "="*80)
    print("Demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    demonstrate_future_work()
