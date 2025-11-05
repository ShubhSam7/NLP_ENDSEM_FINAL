"""
Learned Confidence Model for Sentiment Analysis
Trains a LightGBM model to predict confidence scores based on component analysis
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')


class ConfidenceLearner:
    """
    Learns to predict confidence scores based on sentiment component patterns.

    Uses features like:
    - Component agreement/disagreement
    - Signal strength
    - Variance and spread
    - Component-specific patterns
    """

    def __init__(self):
        self.model = None
        self.metrics = {}
        self.feature_names = []

    def _extract_features(self, component_scores, textblob_subjectivity=0.5,
                         finbert_confidence=0.0, finbert_available=False,
                         has_entities=False):
        """
        Extract features from component scores for confidence prediction.

        Returns feature vector for the model.
        """
        component_values = list(component_scores.values())

        # Basic statistics
        mean_score = np.mean(component_values)
        std_score = np.std(component_values)
        max_score = np.max(np.abs(component_values))
        min_score = np.min(np.abs(component_values))

        # Agreement metrics
        positive_count = sum(1 for x in component_values if x > 0.1)
        negative_count = sum(1 for x in component_values if x < -0.1)
        neutral_count = sum(1 for x in component_values if -0.1 <= x <= 0.1)
        total_count = len(component_values)

        # Agreement ratio (how many components agree on direction)
        agreement_ratio = max(positive_count, negative_count) / total_count

        # Signal strength (average absolute value)
        signal_strength = np.mean(np.abs(component_values))

        # Direction consistency (are all components same sign?)
        same_sign = (all(x >= 0 for x in component_values) or
                    all(x <= 0 for x in component_values))

        # Range of scores
        score_range = max_score - min_score

        # Individual component strengths
        textblob_strength = abs(component_scores.get('textblob', 0))
        finbert_strength = abs(component_scores.get('finbert', 0))
        keyword_strength = abs(component_scores.get('keywords', 0))
        regime_strength = abs(component_scores.get('regime', 0))
        summary_strength = abs(component_scores.get('summary', 0))
        temporal_strength = abs(component_scores.get('temporal', 0))

        # FinBERT specific
        finbert_available_flag = 1.0 if finbert_available else 0.0
        finbert_conf = finbert_confidence if finbert_available else 0.0

        # NER specific
        has_entities_flag = 1.0 if has_entities else 0.0

        # TextBlob objectivity
        objectivity = 1 - textblob_subjectivity

        # Disagreement penalty (coefficient of variation)
        cv = std_score / (abs(mean_score) + 1e-6)  # Coefficient of variation

        # Feature vector
        features = [
            signal_strength,        # 0: Overall signal strength
            std_score,             # 1: Standard deviation (disagreement)
            agreement_ratio,       # 2: Agreement ratio
            same_sign,             # 3: Same sign indicator
            score_range,           # 4: Range of scores
            textblob_strength,     # 5: TextBlob strength
            finbert_strength,      # 6: FinBERT strength
            keyword_strength,      # 7: Keyword strength
            regime_strength,       # 8: Regime strength
            summary_strength,      # 9: Summary strength
            temporal_strength,     # 10: Temporal strength
            finbert_available_flag, # 11: FinBERT available
            finbert_conf,          # 12: FinBERT confidence
            has_entities_flag,     # 13: Has financial entities
            objectivity,           # 14: TextBlob objectivity
            cv,                    # 15: Coefficient of variation
            neutral_count / total_count,  # 16: Neutral component ratio
            max_score,             # 17: Maximum component strength
        ]

        return np.array(features)

    def generate_training_data(self, n_samples=5000):
        """
        Generate synthetic training data for confidence learning.

        Creates diverse scenarios with known confidence levels:
        - High confidence: Strong agreement, high signal
        - Medium confidence: Moderate agreement or signal
        - Low confidence: High disagreement or weak signal
        """
        X = []
        y = []

        np.random.seed(42)

        for i in range(n_samples):
            # Randomly select scenario type
            scenario = np.random.choice(['high_conf', 'medium_conf', 'low_conf', 'very_low_conf'])

            if scenario == 'high_conf':
                # High confidence: Strong agreement + strong signal
                base_sentiment = np.random.uniform(-0.9, 0.9)
                noise = np.random.normal(0, 0.05, 6)  # Low noise
                component_scores = {
                    'textblob': base_sentiment + noise[0],
                    'finbert': base_sentiment + noise[1],
                    'keywords': base_sentiment + noise[2],
                    'regime': base_sentiment * 0.8 + noise[3],
                    'summary': base_sentiment + noise[4],
                    'temporal': base_sentiment * 0.7 + noise[5]
                }
                # High confidence: 0.75 - 0.95
                target_confidence = np.random.uniform(0.75, 0.95)
                finbert_available = np.random.choice([True, False], p=[0.7, 0.3])
                has_entities = np.random.choice([True, False], p=[0.8, 0.2])
                textblob_subj = np.random.uniform(0.1, 0.3)  # Low subjectivity
                finbert_conf = np.random.uniform(0.8, 0.95) if finbert_available else 0

            elif scenario == 'medium_conf':
                # Medium confidence: Moderate agreement or moderate signal
                base_sentiment = np.random.uniform(-0.6, 0.6)
                noise = np.random.normal(0, 0.15, 6)  # Medium noise
                component_scores = {
                    'textblob': base_sentiment + noise[0],
                    'finbert': base_sentiment + noise[1] if np.random.rand() > 0.3 else 0,
                    'keywords': base_sentiment + noise[2],
                    'regime': base_sentiment * 0.6 + noise[3],
                    'summary': base_sentiment + noise[4],
                    'temporal': base_sentiment * 0.5 + noise[5]
                }
                # Medium confidence: 0.45 - 0.75
                target_confidence = np.random.uniform(0.45, 0.75)
                finbert_available = np.random.choice([True, False], p=[0.5, 0.5])
                has_entities = np.random.choice([True, False], p=[0.5, 0.5])
                textblob_subj = np.random.uniform(0.3, 0.6)
                finbert_conf = np.random.uniform(0.5, 0.8) if finbert_available else 0

            elif scenario == 'low_conf':
                # Low confidence: High disagreement or weak signal
                sentiments = np.random.uniform(-0.5, 0.5, 6)  # Random sentiments
                component_scores = {
                    'textblob': sentiments[0],
                    'finbert': sentiments[1] if np.random.rand() > 0.5 else 0,
                    'keywords': sentiments[2],
                    'regime': sentiments[3],
                    'summary': sentiments[4],
                    'temporal': sentiments[5]
                }
                # Low confidence: 0.25 - 0.45
                target_confidence = np.random.uniform(0.25, 0.45)
                finbert_available = np.random.choice([True, False], p=[0.3, 0.7])
                has_entities = np.random.choice([True, False], p=[0.3, 0.7])
                textblob_subj = np.random.uniform(0.5, 0.8)
                finbert_conf = np.random.uniform(0.3, 0.6) if finbert_available else 0

            else:  # very_low_conf
                # Very low confidence: Near zero or highly conflicting
                # Special case: all zeros should have very low confidence
                if np.random.rand() < 0.3:  # 30% chance of all zeros
                    component_scores = {
                        'textblob': 0,
                        'finbert': 0,
                        'keywords': 0,
                        'regime': 0,
                        'summary': 0,
                        'temporal': 0
                    }
                    target_confidence = np.random.uniform(0.05, 0.15)
                else:
                    sentiments = np.random.uniform(-0.3, 0.3, 6)
                    component_scores = {
                        'textblob': sentiments[0],
                        'finbert': 0,  # Often no FinBERT
                        'keywords': sentiments[2],
                        'regime': sentiments[3],
                        'summary': sentiments[4],
                        'temporal': sentiments[5]
                    }
                    target_confidence = np.random.uniform(0.05, 0.25)

                finbert_available = False
                has_entities = False
                textblob_subj = np.random.uniform(0.7, 0.95)
                finbert_conf = 0

            # Clip component scores to valid range
            component_scores = {k: np.clip(v, -1, 1) for k, v in component_scores.items()}

            # Extract features
            features = self._extract_features(
                component_scores,
                textblob_subjectivity=textblob_subj,
                finbert_confidence=finbert_conf,
                finbert_available=finbert_available,
                has_entities=has_entities
            )

            X.append(features)
            y.append(target_confidence)

        return np.array(X), np.array(y)

    def train(self, n_samples=5000):
        """Train the confidence prediction model"""
        print(f"Generating {n_samples} training samples...")
        X, y = self.generate_training_data(n_samples)

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")

        # Define feature names
        self.feature_names = [
            'signal_strength', 'std_score', 'agreement_ratio', 'same_sign',
            'score_range', 'textblob_strength', 'finbert_strength',
            'keyword_strength', 'regime_strength', 'summary_strength',
            'temporal_strength', 'finbert_available', 'finbert_conf',
            'has_entities', 'objectivity', 'cv', 'neutral_ratio', 'max_score'
        ]

        # Train LightGBM model
        print("\nTraining LightGBM Confidence Model...")

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=50)
            ]
        )

        # Evaluate
        print("\n" + "="*80)
        print("EVALUATING CONFIDENCE MODEL")
        print("="*80)

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        # Clip predictions to valid range
        y_train_pred = np.clip(y_train_pred, 0, 1)
        y_val_pred = np.clip(y_val_pred, 0, 1)

        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)

        self.metrics = {
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'train_r2': train_r2,
            'val_r2': val_r2
        }

        print(f"\nTraining Metrics:")
        print(f"  MAE:  {train_mae:.4f}")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R²:   {train_r2:.4f}")

        print(f"\nValidation Metrics:")
        print(f"  MAE:  {val_mae:.4f}")
        print(f"  RMSE: {val_rmse:.4f}")
        print(f"  R²:   {val_r2:.4f}")

        # Feature importance
        print(f"\nTop 10 Feature Importances:")
        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = sorted(zip(self.feature_names, importance),
                                   key=lambda x: x[1], reverse=True)
        for name, imp in feature_importance[:10]:
            print(f"  {name:20s}: {imp:8.1f}")

        print("="*80)

        return self.metrics

    def predict_confidence(self, component_scores, textblob_subjectivity=0.5,
                          finbert_confidence=0.0, finbert_available=False,
                          has_entities=False):
        """
        Predict confidence score for given component analysis.

        Returns confidence score (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Special case: if all components are near zero, return very low confidence
        component_values = list(component_scores.values())
        max_abs = max(abs(v) for v in component_values)
        if max_abs < 0.05:  # All components essentially zero
            return 0.10  # Very low confidence for no signal

        features = self._extract_features(
            component_scores,
            textblob_subjectivity=textblob_subjectivity,
            finbert_confidence=finbert_confidence,
            finbert_available=finbert_available,
            has_entities=has_entities
        )

        confidence = self.model.predict([features])[0]
        return np.clip(confidence, 0, 1)

    def save(self, filepath='confidence_model.pkl'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save. Train first.")

        model_data = {
            'model': self.model,
            'metrics': self.metrics,
            'feature_names': self.feature_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load(self, filepath='confidence_model.pkl'):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.metrics = model_data['metrics']
        self.feature_names = model_data['feature_names']

        print(f"Model loaded from {filepath}")
        print(f"Validation MAE: {self.metrics.get('val_mae', 0):.4f}")
        print(f"Validation R²:  {self.metrics.get('val_r2', 0):.4f}")


if __name__ == "__main__":
    # Train and save the confidence model
    learner = ConfidenceLearner()
    learner.train(n_samples=10000)
    learner.save('confidence_model.pkl')

    print("\n" + "="*80)
    print("TESTING CONFIDENCE PREDICTIONS")
    print("="*80)

    # Test predictions
    test_cases = [
        {
            'name': 'Strong Agreement + High Signal',
            'components': {'textblob': 0.8, 'finbert': 0.85, 'keywords': 0.9,
                          'regime': 0.7, 'summary': 0.8, 'temporal': 0.75},
            'subjectivity': 0.2,
            'finbert_conf': 0.9,
            'finbert_avail': True,
            'entities': True,
            'expected': '>0.80'
        },
        {
            'name': 'High Disagreement',
            'components': {'textblob': 0.5, 'finbert': -0.3, 'keywords': 0.6,
                          'regime': -0.2, 'summary': 0.4, 'temporal': 0.1},
            'subjectivity': 0.5,
            'finbert_conf': 0.4,
            'finbert_avail': True,
            'entities': False,
            'expected': '<0.50'
        },
        {
            'name': 'Weak Signal',
            'components': {'textblob': 0.0, 'finbert': 0.0, 'keywords': 0.05,
                          'regime': 0.0, 'summary': 0.0, 'temporal': 0.0},
            'subjectivity': 0.8,
            'finbert_conf': 0.0,
            'finbert_avail': False,
            'entities': False,
            'expected': '<0.30'
        }
    ]

    for case in test_cases:
        confidence = learner.predict_confidence(
            component_scores=case['components'],
            textblob_subjectivity=case['subjectivity'],
            finbert_confidence=case['finbert_conf'],
            finbert_available=case['finbert_avail'],
            has_entities=case['entities']
        )
        print(f"\n{case['name']}:")
        print(f"  Predicted Confidence: {confidence:.4f}")
        print(f"  Expected: {case['expected']}")
        print(f"  Components: {case['components']}")

    print("\n" + "="*80)
