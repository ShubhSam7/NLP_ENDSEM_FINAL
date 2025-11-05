"""
LightGBM Meta-Learner for Sentiment Ensemble
Combines 6 base sentiment models with learned weights
Includes accuracy, precision, and performance metrics
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)
import joblib
import os


class LightGBMMetaLearner:
    """Meta-learner that combines base sentiment models using LightGBM"""

    def __init__(self, model_path='lightgbm_meta_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.feature_importance = None
        self.metrics = {}

    def prepare_features(self, component_scores, metadata=None):
        """
        Extract features from component scores and metadata

        Args:
            component_scores: dict with keys:
                - textblob, finbert, keywords, regime, summary, temporal
            metadata: optional dict with confidence, date, text_length

        Returns:
            numpy array of shape (7,) with features
        """
        # Base features (6 component scores)
        base_features = [
            component_scores.get('textblob', 0),
            component_scores.get('finbert', 0),
            component_scores.get('keywords', 0),
            component_scores.get('regime', 0),
            component_scores.get('summary', 0),
            component_scores.get('temporal', 0)
        ]

        # Engineered feature: score variance (disagreement measure)
        score_variance = np.std(base_features)

        # Total: 7 features
        features = base_features + [score_variance]

        return np.array(features)

    def train(self, training_data, labels, validation_split=0.2):
        """
        Train the LightGBM meta-learner

        Args:
            training_data: numpy array of shape (n_samples, 7)
            labels: numpy array of shape (n_samples,) with true sentiment scores
            validation_split: fraction of data for validation

        Returns:
            dict with training metrics
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            training_data, labels,
            test_size=validation_split,
            random_state=42
        )

        # LightGBM parameters (optimized for CPU and small datasets)
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.05,
            'n_estimators': 150,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }

        # Train model
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )

        # Calculate metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        # Comprehensive metrics
        self.metrics = {
            # Validation metrics
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'val_r2': r2_score(y_val, y_val_pred),
            'val_accuracy': self._calculate_accuracy(y_val, y_val_pred),
            'val_precision': self._calculate_precision(y_val, y_val_pred),

            # Training metrics
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred),

            # Model info
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_features': X_train.shape[1],
            'n_trees': self.model.n_estimators_
        }

        # Feature importance
        feature_names = ['TextBlob', 'FinBERT', 'Keywords', 'Regime',
                        'Summary', 'Temporal', 'Score_Variance']
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return self.metrics

    def _calculate_accuracy(self, y_true, y_pred, tolerance=0.1):
        """
        Calculate accuracy as % of predictions within tolerance of true value

        For sentiment in [-1, 1], ±0.1 tolerance means predictions
        within 10% of the full range are considered correct
        """
        within_tolerance = np.abs(y_true - y_pred) <= tolerance
        return np.mean(within_tolerance)

    def _calculate_precision(self, y_true, y_pred):
        """
        Calculate precision as consistency of predictions
        Using 1 - (std of residuals / std of true values)
        """
        residuals = y_true - y_pred
        if np.std(y_true) == 0:
            return 1.0
        precision = 1 - (np.std(residuals) / np.std(y_true))
        return max(0, min(1, precision))  # Clip to [0, 1]

    def predict(self, component_scores, metadata=None):
        """
        Predict sentiment using trained meta-learner

        Args:
            component_scores: dict with 6 component scores
            metadata: optional metadata

        Returns:
            float: predicted sentiment in [-1, 1]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() or load() first.")

        features = self.prepare_features(component_scores, metadata)
        prediction = self.model.predict(features.reshape(1, -1))[0]

        # Clip to [-1, 1] range
        return np.clip(prediction, -1, 1)

    def save(self, path=None):
        """Save trained model to disk"""
        if path is None:
            path = self.model_path

        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics
        }

        joblib.dump(model_data, path)
        print(f"Model saved to {path}")

    def load(self, path=None):
        """Load trained model from disk"""
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_importance = model_data.get('feature_importance')
        self.metrics = model_data.get('metrics', {})

        print(f"Model loaded from {path}")
        return self.metrics

    def get_metrics_summary(self):
        """Get formatted metrics summary"""
        if not self.metrics:
            return "No metrics available. Train or load model first."

        summary = f"""
LightGBM Meta-Learner Performance Metrics
==========================================

Validation Performance:
  Accuracy:  {self.metrics.get('val_accuracy', 0):.2%} (within ±0.1 tolerance)
  Precision: {self.metrics.get('val_precision', 0):.3f}
  MAE:       {self.metrics.get('val_mae', 0):.4f}
  RMSE:      {self.metrics.get('val_rmse', 0):.4f}
  R² Score:  {self.metrics.get('val_r2', 0):.4f}

Training Performance:
  MAE:       {self.metrics.get('train_mae', 0):.4f}
  R² Score:  {self.metrics.get('train_r2', 0):.4f}

Model Info:
  Training samples:   {self.metrics.get('n_train_samples', 0)}
  Validation samples: {self.metrics.get('n_val_samples', 0)}
  Features used:      {self.metrics.get('n_features', 0)}
  Trees in ensemble:  {self.metrics.get('n_trees', 0)}

Feature Importance (Top 3):
"""
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(3)
            for _, row in top_features.iterrows():
                summary += f"  {row['feature']:15s}: {row['importance']:.1f}%\n"

        return summary


def generate_synthetic_training_data(n_samples=1000):
    """
    Generate synthetic training data for initial model training

    In production, replace this with actual historical data
    """
    np.random.seed(42)

    # Generate base component scores
    textblob = np.random.uniform(-0.3, 0.3, n_samples)
    finbert = np.random.uniform(-0.5, 0.5, n_samples)
    keywords = np.random.uniform(-0.4, 0.4, n_samples)
    regime = np.random.uniform(-0.3, 0.3, n_samples)
    summary = np.random.uniform(-0.3, 0.3, n_samples)
    temporal = np.random.uniform(-0.2, 0.2, n_samples)

    # Calculate score variance
    all_scores = np.column_stack([textblob, finbert, keywords, regime, summary, temporal])
    score_variance = np.std(all_scores, axis=1)

    # Combine into feature matrix
    X = np.column_stack([textblob, finbert, keywords, regime, summary, temporal, score_variance])

    # Generate labels (weighted combination with some noise)
    # FinBERT has highest weight (0.35), then keywords (0.25), etc.
    weights = np.array([0.12, 0.35, 0.25, 0.15, 0.08, 0.05])
    y = np.dot(all_scores, weights) + np.random.normal(0, 0.05, n_samples)

    # Clip labels to [-1, 1]
    y = np.clip(y, -1, 1)

    return X, y
