"""
Test FinBERT integration with main_app.py FinLlamaAnalyzer
"""

import sys
import os

# Suppress streamlit warnings
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
import warnings
warnings.filterwarnings('ignore')

from main_app import FinLlamaAnalyzer

print("="*80)
print("TESTING FINBERT IN MAIN APP")
print("="*80)

# Initialize analyzer
print("\nInitializing FinLlamaAnalyzer...")
analyzer = FinLlamaAnalyzer(api_key="test_key")

print(f"\nFinBERT Available: {analyzer.finbert_available}")
print(f"NER Available: {analyzer.ner_available}")
print(f"Meta-Learner Available: {analyzer.use_meta_learner}")
print(f"Confidence Learner Available: {analyzer.use_confidence_learner}")

# Test cases
test_cases = [
    "Apple reported record quarterly earnings of $25.5 billion, beating expectations by 15%.",
    "The company filed for bankruptcy after posting massive losses.",
    "Markets rallied on strong economic data and positive sentiment.",
]

print("\n" + "="*80)
print("TESTING SENTIMENT ANALYSIS WITH FINBERT")
print("="*80)

for i, text in enumerate(test_cases, 1):
    print(f"\n[Test {i}] {text[:60]}...")
    result = analyzer.finllama_ensemble_analysis(text)

    print(f"  Ensemble Score:  {result['ensemble_score']:+.4f}")
    print(f"  Confidence:      {result['confidence']:.4f}")
    print(f"  Method:          {result['method']}")
    print(f"  Components:")
    for comp, score in result['components'].items():
        print(f"    {comp:12s}: {score:+.4f}")

print("\n" + "="*80)
if analyzer.finbert_available:
    print("SUCCESS: FINBERT IS WORKING IN MAIN APP!")
    print("Check that finbert scores are now NON-ZERO above")
else:
    print("WARNING: FinBERT still not loading in main app")
print("="*80)
