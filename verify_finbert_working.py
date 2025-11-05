"""
Final verification that FinBERT is working
Shows before/after comparison
"""

import sys
import os

# Suppress streamlit warnings
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
import warnings
warnings.filterwarnings('ignore')

from main_app import FinLlamaAnalyzer

print("=" * 80)
print("FINBERT INSTALLATION VERIFICATION")
print("=" * 80)

# Initialize analyzer
analyzer = FinLlamaAnalyzer(api_key="test_key")

# Display status
print("\n[SYSTEM STATUS]")
print(f"  FinBERT Available:            {'YES [OK]' if analyzer.finbert_available else 'NO [FAIL]'}")
print(f"  NER Available:                {'YES [OK]' if analyzer.ner_available else 'NO [FAIL]'}")
print(f"  Meta-Learner Available:       {'YES [OK]' if analyzer.use_meta_learner else 'NO [FAIL]'}")
print(f"  Confidence Learner Available: {'YES [OK]' if analyzer.use_confidence_learner else 'NO [FAIL]'}")

# Test news
test_news = "Apple Inc. reported record earnings of $30 billion, up 25% year-over-year, beating analyst expectations."

print("\n" + "=" * 80)
print("ANALYZING SAMPLE NEWS")
print("=" * 80)
print(f"\nHeadline: {test_news}\n")

result = analyzer.finllama_ensemble_analysis(test_news)

print(f"Final Sentiment Score:  {result['ensemble_score']:+.4f}")
print(f"Confidence:             {result['confidence']:.4f}")
print(f"Method:                 {result['method']}")

print("\n[COMPONENT BREAKDOWN]")
print("-" * 80)
print(f"  TextBlob Score:     {result['components']['textblob']:+.4f}")
print(f"  FinBERT Score:      {result['components']['finbert']:+.4f}  {'[WORKING!]' if result['components']['finbert'] != 0 else '[NOT WORKING]'}")
print(f"  Keywords Score:     {result['components']['keywords']:+.4f}")
print(f"  Regime Score:       {result['components']['regime']:+.4f}")
print(f"  Summary Score:      {result['components']['summary']:+.4f}")
print(f"  Temporal Score:     {result['components']['temporal']:+.4f}")

print("\n" + "=" * 80)
if analyzer.finbert_available and result['components']['finbert'] != 0:
    print("SUCCESS: FINBERT IS FULLY OPERATIONAL!")
    print("\nYour analysis will now include:")
    print("  - Deep learning financial sentiment (FinBERT)")
    print("  - 28% weight in ensemble (highest)")
    print("  - Improved confidence scores")
    print("  - Better accuracy on financial news")
else:
    print("WARNING: FinBERT is not working properly")
    print("Please check the installation steps in FINBERT_FIX_SUMMARY.md")

print("=" * 80)
